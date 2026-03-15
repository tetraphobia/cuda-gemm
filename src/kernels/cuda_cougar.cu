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
          const uint TILE_M, const uint TILE_N>
__global__ void _cougar_kernel(float alpha, float *A, float *B, float beta,
                               float *C, int m, int k, int n) {
  __shared__ float shared_A[BLOCK_M * BLOCK_K];
  __shared__ float shared_B[BLOCK_K * BLOCK_N];

  // Block-level tile origin
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const uint threadsPerRow = BLOCK_N / TILE_N;

  const uint threadCol = threadIdx.x % threadsPerRow;
  const uint threadRow = threadIdx.x / threadsPerRow;

  const uint numThreads = (BLOCK_M / TILE_M) * (BLOCK_N / TILE_N);
  const uint innerRowA = threadIdx.x / BLOCK_K;
  const uint innerColA = threadIdx.x % BLOCK_K;
  const uint innerRowB = threadIdx.x / BLOCK_N;
  const uint innerColB = threadIdx.x % BLOCK_N;

  // How many rows each load iteration covers
  const uint strideA = numThreads / BLOCK_K;
  const uint strideB = numThreads / BLOCK_N;

  // Set pointers to starting positions
  A += cRow * BLOCK_M * k;
  B += cCol * BLOCK_N;
  C += cRow * BLOCK_M * n + cCol * BLOCK_N;

  // 2D register accumulator
  float acc[TILE_M][TILE_N] = {{0.0f}};

  for (int block = 0; block < k; block += BLOCK_K) {
    // Coalesce-load shared_A
    for (uint offset = 0; offset < BLOCK_M; offset += strideA) {
      int aRow = cRow * BLOCK_M + innerRowA + offset;
      int aCol = block + innerColA;
      shared_A[(innerRowA + offset) * BLOCK_K + innerColA] =
          (aRow < m && aCol < k) ? A[(innerRowA + offset) * k + innerColA]
                                 : 0.0f;
    }

    // Coalesce-load shared_B
    for (uint offset = 0; offset < BLOCK_K; offset += strideB) {
      int bRow = block + innerRowB + offset;
      int bCol = cCol * BLOCK_N + innerColB;
      shared_B[(innerRowB + offset) * BLOCK_N + innerColB] =
          (bRow < k && bCol < n) ? B[(innerRowB + offset) * n + innerColB]
                                 : 0.0f;
    }

    __syncthreads();
    A += BLOCK_K;
    B += BLOCK_K * n;

    // Compute per-thread results
    for (uint i = 0; i < BLOCK_K; i++) {
      // Cache A and B into registers
      float regA[TILE_M], regB[TILE_N];

      for (uint rm = 0; rm < TILE_M; rm++) {
        regA[rm] = shared_A[(threadRow * TILE_M + rm) * BLOCK_K + i];
      }

      for (uint rn = 0; rn < TILE_N; rn++) {
        regB[rn] = shared_B[i * BLOCK_N + threadCol * TILE_N + rn];
      }

      // Accumulate outer product of regA and regB
      for (uint rn = 0; rn < TILE_N; rn++) {
        float tmpB = regB[rn];
        for (uint rm = 0; rm < TILE_M; rm++) {
          acc[rm][rn] += regA[rm] * tmpB;
        }
      }
    }

    __syncthreads();
  }

  // Write TILE_M x TILE_N results to output matrix
  for (uint rm = 0; rm < TILE_M; rm++) {
    int outRow = cRow * BLOCK_M + threadRow * TILE_M + rm;
    for (uint rn = 0; rn < TILE_N; rn++) {
      int outCol = cCol * BLOCK_N + threadCol * TILE_N + rn;
      if (outRow < m && outCol < n) {
        C[(threadRow * TILE_M + rm) * n + threadCol * TILE_N + rn] =
            alpha * acc[rm][rn] +
            beta * C[(threadRow * TILE_M + rm) * n + threadCol * TILE_N + rn];
      }
    }
  }
}

void multiply_cougar(float alpha, const float *A, const float *B, float beta,
                     float *C, int m, int k, int n, kernel_args_t *args) {
  const uint BLOCK_M = 128;
  const uint BLOCK_N = 128;
  const uint BLOCK_K = 16;
  const uint TILE_M = 8;
  const uint TILE_N = 8;
  dim3 blockDim((BLOCK_M / TILE_M) * (BLOCK_N / TILE_N));
  dim3 gridDim(CEIL_DIV(m, BLOCK_M), CEIL_DIV(n, BLOCK_N));

  float *A_ptr = (float *)A;
  float *B_ptr = (float *)B;
  float *C_ptr = (float *)C;

  if (args->stream)
    _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M, TILE_N>
        <<<gridDim, blockDim, 0, args->stream>>>(alpha, A_ptr, B_ptr, beta,
                                                 C_ptr, m, k, n);
  else
    _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M, TILE_N>
        <<<gridDim, blockDim>>>(alpha, A_ptr, B_ptr, beta, C_ptr, m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("_cougar_kernel launch error: %s\n", cudaGetErrorString(err));
}

void multiply_cougar(const float *A, const float *B, float *C, int m, int k,
                     int n, kernel_args_t *args) {
  multiply_cougar(ALPHA_DEFAULT, A, B, BETA_DEFAULT, C, m, k, n, args);
}
