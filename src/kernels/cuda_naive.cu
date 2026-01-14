#include "cuda_naive.cuh"

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
__global__ void multiply_cuda_naive(const float *A, const float *B, float *C,
                                    int m, int k, int n) {
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
