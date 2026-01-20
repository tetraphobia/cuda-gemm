#include "cuda_rf_block.cuh"

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

#define TM 8 // rows per thread
#define TN 8 // columns per thread

/**
 * Multiply two matrices `A` and `B` using CUDA with shared
 * memory tiling and register file sub-blocking
 * and store result in matrix `C`.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 * Matrix `A` should have `m` rows and `k` columns.
 * Matrix `B` should have `k` rows and `n` columns.
 * Resulting matrix `C` should have `m` rows and `n` columns.
 *
 */
__global__ void multiply_cuda_rf_block(const float * A,
                                       const float * B,
                                       float * C, int m, int k,
                                       int n) {
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  // Thread indices within block
  int tx = threadIdx.x; // [0, TILE_N/TN)
  int ty = threadIdx.y; // [0, TILE_M/TM)

  // Global tile origin
  int blockRow = blockIdx.y * TILE_M;
  int blockCol = blockIdx.x * TILE_N;

  // Per-thread output tile origin
  int rowBase = blockRow + ty * TM;
  int colBase = blockCol + tx * TN;

  // Register accumulation
  float acc[TM][TN];
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  // Loop over K tiles
  for (int kt = 0; kt < k; kt += TILE_K) {

    // Cooperative load A tile
    for (int i = 0; i < TM; ++i) {
      int r = rowBase + i;
      int c = kt + tx;
      As[ty * TM + i][tx] = (r < m && c < k) ? A[r * k + c] : 0.0f;
    }

    // Cooperative load B tile
    for (int j = 0; j < TN; ++j) {
      int r = kt + ty;
      int c = colBase + j;
      Bs[ty][tx * TN + j] = (r < k && c < n) ? B[r * n + c] : 0.0f;
    }

    __syncthreads();

    // Compute register block
    for (int k = 0; k < TILE_K; ++k) {
      float aReg[TM];
      for (int i = 0; i < TM; ++i)
        aReg[i] = As[ty * TM + i][k];

      for (int j = 0; j < TN; ++j) {
        float b = Bs[k][tx * TN + j];
        for (int i = 0; i < TM; ++i)
          acc[i][j] += aReg[i] * b;
      }
    }

    __syncthreads();
  }

  // Write results
  for (int i = 0; i < TM; ++i) {
    int r = rowBase + i;
    if (r < m) {
      for (int j = 0; j < TN; ++j) {
        int c = colBase + j;
        if (c < n)
          C[r * n + c] = acc[i][j];
      }
    }
  }
}
