#include "cuda_shuffle.h"
#include "stdio.h"

#define TILE_M 64
#define TILE_K 32
#define TILE_N 64

/**
 * Multiply two matrices `A` and `B` using CUDA with shared
 * memory tiling and intra-warp shuffling.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 * This kernel was generated using GPT-5
 */
__global__ void _warp_shuffle_kernel(const float *A, const float *B, float *C,
                                     int m, int k, int n) {
  extern __shared__ float
      smem[]; // size: (TILE_M*TILE_K + TILE_K*TILE_N)*2 floats
  float *As0 = smem;
  float *Bs0 = As0 + TILE_M * TILE_K;
  float *As1 = Bs0 + TILE_K * TILE_N;
  float *Bs1 = As1 + TILE_M * TILE_K;

  const int lane = threadIdx.x & 31;   // 0..31
  const int warpId = threadIdx.x >> 5; // 0..7 (8 warps/block)

  // Map 8 warps as 4 (rows) x 2 (cols) warp tiles → each warp does 32x64
  const int warpRow = warpId / 2; // 0..3
  const int warpCol = warpId % 2; // 0..1

  const int blockRow0 = blockIdx.y * TILE_M;
  const int blockCol0 = blockIdx.x * TILE_N;

  // Warp output origin within the 128x128 C tile
  const int warpRow0 = blockRow0 + warpRow * 32;
  const int warpCol0 = blockCol0 + warpCol * 64;

  // Per-thread row within the 32-row warp tile
  const int row = warpRow0 + lane;

  // 8 subtiles of width 8 to cover 64 output columns per warp
  float acc[8][8]; // [which 8-wide subtile][column within 8]
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      acc[t][j] = 0.f;
  }

  // Number of K tiles
  const int KTILES = (k + TILE_K - 1) / TILE_K;

  // Loader lambda: cooperative global->shared copy for A and B tiles
  auto loadAB = [&](int kt, float *As, float *Bs) {
    // A tile: (TILE_M x TILE_K), base at (blockRow0, kt*TILE_K)
    // B tile: (TILE_K x TILE_N), base at (kt*TILE_K, blockCol0)
    const int t = threadIdx.x; // 0..255
    // Copy A tile
    for (int idx = t; idx < TILE_M * TILE_K; idx += blockDim.x) {
      int i = idx / TILE_K;
      int k = idx % TILE_K;
      int gRow = blockRow0 + i;
      int gCol = kt * TILE_K + k;
      As[idx] = (gRow < m && gCol < k) ? A[(size_t)gRow * k + gCol] : 0.f;
    }
    // Copy B tile
    for (int idx = t; idx < TILE_K * TILE_N; idx += blockDim.x) {
      int k = idx / TILE_N;
      int j = idx % TILE_N;
      int gRow = kt * TILE_K + k;
      int gCol = blockCol0 + j;
      Bs[idx] = (gRow < k && gCol < n) ? B[(size_t)gRow * n + gCol] : 0.f;
    }
  };

  // Preload tile 0
  if (KTILES > 0)
    loadAB(0, As0, Bs0);
  __syncthreads();

  // Iterate K tiles with simple ping-pong buffering
  for (int kt = 0; kt < KTILES; ++kt) {
    // Select stage buffers
    float *As = (kt & 1) ? As1 : As0;
    float *Bs = (kt & 1) ? Bs1 : Bs0;

    // Preload next tile while we compute
    if (kt + 1 < KTILES) {
      float *Asn = (kt & 1) ? As0 : As1;
      float *Bsn = (kt & 1) ? Bs0 : Bs1;
      // Overlap *a bit*: do the preload before we finish; guarded by
      // __syncthreads later.
      loadAB(kt + 1, Asn, Bsn);
    }

    // Compute on this tile: iterate kk in 0..TILE_K-1
#pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      // Load A(row, kk) from shared (guard row)
      float a = 0.f;
      if (row < m)
        a = As[(row - blockRow0) * TILE_K + kk];

      // We process 8 contiguous 8-wide subtiles to cover the warp's 64 columns.
#pragma unroll
      for (int subt = 0; subt < 8; ++subt) {
        // Subtile j-range in global N:
        const int colBase = warpCol0 + subt * 8;

        // Each lane starts on its own column within the subtile: lane%8
        int colOff = lane & 7; // 0..7
        int col = colBase + colOff;

        // Each lane fetches one B(kk, col) from shared, then rotates across
        // warp (only within 8-wide window)
        float b = 0.f;
        if (col < n) {
          // B is stored as TILE_K x TILE_N row-major in Bs
          b = Bs[kk * TILE_N + (col - blockCol0)];
        }

        // Do a circular 8-step rotation: accumulate into acc[subt][*]
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          if (row < m && col < n) {
            acc[subt][colOff] += a * b;
          }
          // Rotate b one lane to the right inside the warp ring
          b = __shfl_sync(0xffffffffu, b, (lane + 31) & 31);
          // Advance index within the 8-wide subtile
          colOff = (colOff + 1) & 7;
          col = colBase + colOff;
        }
      }
    }

    __syncthreads(); // ensure next stage is fully written before swapping
  }

  // Write out the 32x64 results for this warp
  if (row < m) {
#pragma unroll
    for (int subt = 0; subt < 8; ++subt) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int col = warpCol0 + subt * 8 + j;
        if (col < n) {
          C[(size_t)row * n + col] = acc[subt][j];
        }
      }
    }
  }
}

void multiply_warp_shuffle(const float *A, const float *B, float *C, int m,
                           int k, int n, kernel_args_t *args) {
  dim3 block(256);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  size_t smem_bytes = 2 * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);

  if (args->stream != 0)
    _warp_shuffle_kernel<<<grid, block, smem_bytes, args->stream>>>(A, B, C, m, k, n);
  else
    _warp_shuffle_kernel<<<grid, block, smem_bytes>>>(A, B, C, m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Shared kernel launch error: %s\n", cudaGetErrorString(err));
}
