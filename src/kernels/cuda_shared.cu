#include "cuda_shared.h"
#include "stdio.h"

#define TILE_M 16
#define TILE_K 16
#define TILE_N 16

/**
 * Multiply two matrices `A` and `B` using CUDA with shared
 * memory tiling and store result in matrix `C`.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 */
__global__ void _shared_kernel(const float *A, const float *B, float *C, int m,
                               int k, int n) {
  __shared__ float shared_A[TILE_M][TILE_K];
  __shared__ float shared_B[TILE_K][TILE_N];

  int globalRow = blockIdx.y * TILE_M + threadIdx.y;
  int globalCol = blockIdx.x * TILE_N + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (k + TILE_K - 1) / TILE_K; t++) {

    // Load both tiles into shared memory
    int aRow = globalRow;
    int aCol = t * TILE_K + col;
    shared_A[row][col] = (aRow < m && aCol < k) ? A[aRow * k + aCol] : 0.0f;

    int bRow = t * TILE_K + row;
    int bCol = globalCol;
    shared_B[row][col] = (bRow < k && bCol < n) ? B[bRow * n + bCol] : 0.0f;

    __syncthreads();

    // Compute partial product for this tile
    for (int i = 0; i < TILE_K; i++)
      sum += shared_A[row][i] * shared_B[i][col];

    __syncthreads();
  }

  // Write the result to global memory
  if (globalRow < m && globalCol < n)
    C[globalRow * n + globalCol] = sum;
}

void multiply_cuda_shared(const float *A, const float *B, float *C, int m,
                          int k, int n, kernel_args_t *args) {
  dim3 block(TILE_N, TILE_M);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  if (args->stream != 0)
    _shared_kernel<<<grid, block, 0, args->stream>>>(A, B, C, m, k, n);
  else
    _shared_kernel<<<grid, block>>>(A, B, C, m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Shared kernel launch error: %s\n", cudaGetErrorString(err));
}
