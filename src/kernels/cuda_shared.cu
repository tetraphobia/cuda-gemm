#include "cuda_shared.cuh"

#define TILE_SIZE 16

/**
 * Multiply two matrices `A` and `B` using CUDA with shared
 * memory tiling and store result in matrix `C`.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 * Matrix `A` should have `m` rows and `k` columns.
 * Matrix `B` should have `k` rows and `n` columns.
 * Resulting matrix `C` should have `m` rows and `n` columns.
 *
 * Adapted from https://kharshit.github.io/blog/2024/06/07/matrix-multiplication-cuda
 */
__global__ void multiply_cuda_shared(const float *A, const float *B, float *C, int m, int k,
                                     int n) {
  __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

  int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
  int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  float sum = 0.0f;

  for (int i = 0; i < (k + TILE_SIZE - 1) / TILE_SIZE; i++) {

    // Load A tile into shared memory
    if (globalRow < m && (i * TILE_SIZE + col) < k)
      shared_A[row][col] = A[globalRow * k + i * TILE_SIZE + col];
    else
      shared_A[row][col] = 0.0f;

    // Load B tile into shared memory
    if ((i * TILE_SIZE + row) < k && globalCol < n)
      shared_B[row][col] = B[(i * TILE_SIZE + row) * n + globalCol];
    else
      shared_B[row][col] = 0.0f;

    __syncthreads();

    // Compute partial product
    for (int j = 0; j < TILE_SIZE; j++)
      sum += shared_A[row][j] * shared_B[j][col];

    __syncthreads();
  }

  // Write result to global memory
  if (globalRow < m && globalCol < n)
    C[globalRow * n + globalCol] = sum;
}
