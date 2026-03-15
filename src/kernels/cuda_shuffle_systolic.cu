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
__global__ void _gemm_systolic_shfl32x8(
    // Each warp computes a 32x32 C tile, but accumulates it as four 32x8
    // subtiles lowering per-thread registers and shuffle pressure.
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int K, int N) {
  int lane = threadIdx.x;   // 0..31
  int warpId = threadIdx.y; // 0..3
  int warpRow = warpId / 2; // 0..1
  int warpCol = warpId % 2; // 0..1

  const int tileRows = 32, tileCols = 32;
  int tileRow0 = (blockIdx.y * 2 + warpRow) * tileRows;
  int tileCol0 = (blockIdx.x * 2 + warpCol) * tileCols;

  int row = tileRow0 + lane;
  unsigned mask = 0xffffffffu;

  // 4 subtiles of width 8
  float acc0[8], acc1[8], acc2[8], acc3[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc0[i] = acc1[i] = acc2[i] = acc3[i] = 0.0f;
  }

  for (int k = 0; k < K; ++k) {
    float a = (row < M) ? A[row * (size_t)K + k] : 0.0f;

    // For each 8-column subtile, load its B element at (k, col)
#pragma unroll 4
    for (int blk = 0; blk < 4; ++blk) {
      int c0 = tileCol0 + blk * 8 +
               lane; // lane selects starting column within the 8-wide ring
      float b = (c0 < N) ? B[k * (size_t)N + c0] : 0.0f;

      // Do an 8-step circular rotation within the warp; only columns within
      // this 8-wide segment are valid
      int colIdx = c0;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        int off = (colIdx - (tileCol0 + blk * 8)) & 7; // 0..7
        if (row < M && colIdx < N) {
          // Select the right accumulator array by blk
          if (blk == 0)
            acc0[off] += a * b;
          else if (blk == 1)
            acc1[off] += a * b;
          else if (blk == 2)
            acc2[off] += a * b;
          else
            acc3[off] += a * b;
        }
        b = __shfl_sync(mask, b, (lane + 31) & 31); // rotate by 1
        colIdx = tileCol0 + blk * 8 +
                 ((off + 1) & 7); // next column inside this 8-wide segment
      }
    }
  }

  // Write back
  if (row < M) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int c = tileCol0 + 0 * 8 + j;
      if (c < N)
        C[row * (size_t)N + c] = acc0[j];
      c = tileCol0 + 1 * 8 + j;
      if (c < N)
        C[row * (size_t)N + c] = acc1[j];
      c = tileCol0 + 2 * 8 + j;
      if (c < N)
        C[row * (size_t)N + c] = acc2[j];
      c = tileCol0 + 3 * 8 + j;
      if (c < N)
        C[row * (size_t)N + c] = acc3[j];
    }
  }
}

void multiply_warp_shuffle_systolic(const float *A, const float *B, float *C,
                                    int m, int k, int n, kernel_args_t *args) {
  dim3 block(256);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  size_t smem_bytes = 2 * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);

  if (args->stream != 0)
    _gemm_systolic_shfl32x8<<<grid, block, smem_bytes, args->stream>>>(A, B, C,
                                                                       m, k, n);
  else
    _gemm_systolic_shfl32x8<<<grid, block, smem_bytes>>>(A, B, C, m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Shared kernel launch error: %s\n", cudaGetErrorString(err));
}
