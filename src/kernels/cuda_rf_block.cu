#include "cuda_rf_block.h"
#include "types.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

#define THREADS_PER_ROW 8 // rows per thread
#define THREADS_PER_COL 8 // columns per thread

/**
 * Multiply two matrices `A` and `B` using CUDA with shared
 * memory tiling and register file sub-blocking
 * and store result in matrix `C`.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 */
__global__ void _rf_block_kernel(const float *A, const float *B, float *C,
                                 int m, int k, int n) {
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];
  // TODO: This needs another pass to make sure everything is correct.

  // Thread indices within block
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;

  // Global tile origin
  int blockRow = blockIdx.y * TILE_M;
  int blockCol = blockIdx.x * TILE_N;

  // Per-thread output tile origin
  int rowBase = blockRow + thread_y * THREADS_PER_ROW;
  int colBase = blockCol + thread_x * THREADS_PER_COL;

  // Register accumulation
  float acc[THREADS_PER_ROW][THREADS_PER_COL];
  for (int i = 0; i < THREADS_PER_ROW; ++i)
    for (int j = 0; j < THREADS_PER_COL; ++j)
      acc[i][j] = 0.0f;

  // Loop over K tiles
  for (int kt = 0; kt < k; kt += TILE_K) {

    // Cooperative load A tile
    for (int i = 0; i < THREADS_PER_ROW; ++i) {
      int r = rowBase + i;
      int c = kt + thread_x;
      As[thread_y * THREADS_PER_ROW + i][thread_x] =
          (r < m && c < k) ? A[r * k + c] : 0.0f;
    }

    // Cooperative load B tile
    for (int j = 0; j < THREADS_PER_COL; ++j) {
      int r = kt + thread_y;
      int c = colBase + j;
      Bs[thread_y][thread_x * THREADS_PER_COL + j] =
          (r < k && c < n) ? B[r * n + c] : 0.0f;
    }

    __syncthreads();

    // Compute register block
    for (int k = 0; k < TILE_K; ++k) {
      float aReg[THREADS_PER_ROW];
      for (int i = 0; i < THREADS_PER_ROW; ++i)
        aReg[i] = As[thread_y * THREADS_PER_ROW + i][k];

      for (int j = 0; j < THREADS_PER_COL; ++j) {
        float b = Bs[k][thread_x * THREADS_PER_COL + j];
        for (int i = 0; i < THREADS_PER_ROW; ++i)
          acc[i][j] += aReg[i] * b;
      }
    }

    __syncthreads();
  }

  // Write results
  for (int i = 0; i < THREADS_PER_ROW; ++i) {
    int r = rowBase + i;
    if (r < m) {
      for (int j = 0; j < THREADS_PER_COL; ++j) {
        int c = colBase + j;
        if (c < n)
          C[r * n + c] = acc[i][j];
      }
    }
  }
}

void multiply_cuda_rf_block(const float *A, const float *B, float *C, int m,
                            int k, int n, kernel_args_t *args) {

  dim3 block(TILE_M / THREADS_PER_ROW, TILE_N / THREADS_PER_COL);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  if (args->stream != NULL) {
    _rf_block_kernel<<<grid, block, 0, args->stream>>>(A, B, C, m, k, n);
  } else {
    _rf_block_kernel<<<grid, block, 0>>>(A, B, C, m, k, n);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
