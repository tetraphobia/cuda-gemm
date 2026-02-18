#include "types.h"
#include <stdio.h>

#define TILE_M 16
#define TILE_N 16

/**
 * Multiply two matrices `A` and `B` using CUDA
 * and store result in matrix `C`.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 * Matrix `A` should have `m` rows and `k` columns.
 * Matrix `B` should have `k` rows and `n` columns.
 * Resulting matrix `C` should have `m` rows and `n` columns.
 */
__global__ void _naive_kernel(const float *A, const float *B, float *C, int m,
                              int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;

    for (int i = 0; i < k; i++) {
      sum += A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = sum;
  }
}

void multiply_cuda_naive(const float *A, const float *B, float *C, int m, int k,
                         int n, kernel_args_t *args) {

  dim3 block(TILE_M, TILE_N);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  if (args->stream != NULL) {
    _naive_kernel<<<grid, block, 0, args->stream>>>(A, B, C, m, k, n);
  } else {
    _naive_kernel<<<grid, block, 0>>>(A, B, C, m, k, n);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
