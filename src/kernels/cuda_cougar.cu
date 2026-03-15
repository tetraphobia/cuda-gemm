#include "cuda_cougar.h"
#include "types.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

#define ALPHA_DEFAULT 1.0f
#define BETA_DEFAULT 0.0f

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/**
 * Multiply two matrices `A` and `B` using CUDA.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 * Based on the work of Simon Boehm
 * https://siboehm.com/articles/22/CUDA-MMM
 */

template <const uint BLOCK_M, const uint BLOCK_K, const uint BLOCK_N,
          const uint TILE_M>
__global__ void _cougar_kernel(float alpha, float *A, float *B, float beta,
                               float *C, int m, int k, int n) {
  __shared__ float shared_A[BLOCK_M * BLOCK_K];
  __shared__ float shared_B[BLOCK_K * BLOCK_N];

  // Row/col of output matrix
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // Thread mapping for computation: each thread computes TILE_M rows
  const uint threadCol = threadIdx.x % BLOCK_N;
  const uint threadRow = threadIdx.x / BLOCK_N;

  // Separate load indices for A (BLOCK_M x BLOCK_K) and B (BLOCK_K x BLOCK_N)
  const uint numThreads = (BLOCK_M * BLOCK_N) / TILE_M;
  const uint innerRowA = threadIdx.x / BLOCK_K;
  const uint innerColA = threadIdx.x % BLOCK_K;
  const uint innerRowB = threadIdx.x / BLOCK_N;
  const uint innerColB = threadIdx.x % BLOCK_N;

  // Set pointers to starting positions
  A += cRow * BLOCK_M * k;
  B += cCol * BLOCK_N;
  C += cRow * BLOCK_M * n + cCol * BLOCK_N;

  float threadSums[TILE_M] = {0.0f};

  for (int block = 0; block < k; block += BLOCK_K) {
    // Coalesce-load elements of A and B into shared memory tiles.
    // Bounds-check: zero-pad when indices fall outside the actual matrix.
    int aRow = cRow * BLOCK_M + innerRowA;
    int aCol = block + innerColA;
    shared_A[innerRowA * BLOCK_K + innerColA] =
        (aRow < m && aCol < k) ? A[innerRowA * k + innerColA] : 0.0f;

    int bRow = block + innerRowB;
    int bCol = cCol * BLOCK_N + innerColB;
    shared_B[innerRowB * BLOCK_N + innerColB] =
        (bRow < k && bCol < n) ? B[innerRowB * n + innerColB] : 0.0f;

    __syncthreads();
    // Shift block tile
    A += BLOCK_K;
    B += BLOCK_K * n;

    // Compute per-thread results
    for (uint i = 0; i < BLOCK_K; i++) {
      float tmpB = shared_B[i * BLOCK_N + threadCol];
      for (uint j = 0; j < TILE_M; j++) {
        threadSums[j] +=
            shared_A[(threadRow * TILE_M + j) * BLOCK_K + i] * tmpB;
      }
    }

    __syncthreads();
  }

  // Write to output matrix (bounds-check for non-tile-aligned edges)
  for (uint i = 0; i < TILE_M; i++) {
    int outRow = cRow * BLOCK_M + threadRow * TILE_M + i;
    int outCol = cCol * BLOCK_N + threadCol;
    if (outRow < m && outCol < n) {
      C[(threadRow * TILE_M + i) * n + threadCol] =
          alpha * threadSums[i] +
          beta * C[(threadRow * TILE_M + i) * n + threadCol];
    }
  }
}

void multiply_cougar(float alpha, const float *A, const float *B, float beta,
                     float *C, int m, int k, int n, kernel_args_t *args) {
  const uint BLOCK_M = 64;
  const uint BLOCK_N = 64;
  const uint BLOCK_K = 8;
  const uint TILE_M = 8;
  dim3 blockDim((BLOCK_M * BLOCK_N) / TILE_M);
  dim3 gridDim(CEIL_DIV(m, BLOCK_M), CEIL_DIV(n, BLOCK_N));

  float *A_ptr = (float *)A;
  float *B_ptr = (float *)B;
  float *C_ptr = (float *)C;

  if (args->stream)
    _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M>
        <<<gridDim, blockDim, 0, args->stream>>>(alpha, A_ptr, B_ptr, beta,
                                                 C_ptr, m, k, n);
  else
    _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M>
        <<<gridDim, blockDim>>>(alpha, A_ptr, B_ptr, beta, C_ptr, m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("_cougar_kernel launch error: %s\n", cudaGetErrorString(err));
}

void multiply_cougar(const float *A, const float *B, float *C, int m, int k,
                     int n, kernel_args_t *args) {
  multiply_cougar(ALPHA_DEFAULT, A, B, BETA_DEFAULT, C, m, k, n, args);
}
