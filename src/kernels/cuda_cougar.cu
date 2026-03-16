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
          const uint TILE_M, const uint TILE_N, bool ALIGNED>
__global__ void _cougar_kernel(float alpha, const float *A, const float *B, float beta,
                               float *C, int m, int k, int n) {
  __shared__ __align__(16) float shared_A[BLOCK_M * BLOCK_K];
  __shared__ __align__(16) float shared_B[BLOCK_K * BLOCK_N];

  // Block-level tile origin
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const uint threadsPerRow = BLOCK_N / TILE_N;
  const uint threadCol = threadIdx.x % threadsPerRow;
  const uint threadRow = threadIdx.x / threadsPerRow;

  const uint numThreads = (BLOCK_M / TILE_M) * (BLOCK_N / TILE_N);

  // float4 load indices for A (BLOCK_M x BLOCK_K)
  // Each thread loads 4 contiguous floats along the K dimension
  const uint innerRowA = threadIdx.x / (BLOCK_K / 4);
  const uint innerColA = threadIdx.x % (BLOCK_K / 4);
  const uint strideA = numThreads / (BLOCK_K / 4);

  // float4 load indices for B (BLOCK_K x BLOCK_N)
  // Each thread loads 4 contiguous floats along the N dimension
  const uint innerRowB = threadIdx.x / (BLOCK_N / 4);
  const uint innerColB = threadIdx.x % (BLOCK_N / 4);
  const uint strideB = numThreads / (BLOCK_N / 4);

  // Set pointers to starting positions
  A += cRow * BLOCK_M * k;
  B += cCol * BLOCK_N;
  C += cRow * BLOCK_M * n + cCol * BLOCK_N;

  // 2D register accumulator
  float acc[TILE_M][TILE_N] = {{0.0f}};

  const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  for (int block = 0; block < k; block += BLOCK_K) {
    // Vectorized coalesce-load shared_A (float4 along K)
    for (uint offset = 0; offset < BLOCK_M; offset += strideA) {
      uint row = innerRowA + offset;
      int aRow = cRow * BLOCK_M + row;
      int aCol = innerColA * 4;
      float4 tmp = zero4;
      if (aRow < m) {
        if (ALIGNED) {
          tmp = reinterpret_cast<const float4 *>(&A[row * k + aCol])[0];
        } else {
          tmp.x = (aCol + 0 < k) ? A[row * k + aCol + 0] : 0.0f;
          tmp.y = (aCol + 1 < k) ? A[row * k + aCol + 1] : 0.0f;
          tmp.z = (aCol + 2 < k) ? A[row * k + aCol + 2] : 0.0f;
          tmp.w = (aCol + 3 < k) ? A[row * k + aCol + 3] : 0.0f;
        }
      }
      reinterpret_cast<float4 *>(&shared_A[row * BLOCK_K + aCol])[0] = tmp;
    }

    // Vectorized coalesce-load shared_B (float4 along N)
    for (uint offset = 0; offset < BLOCK_K; offset += strideB) {
      uint row = innerRowB + offset;
      int bRow = block + row;
      int bCol = innerColB * 4;
      float4 tmp = zero4;
      if (bRow < k) {
        if (ALIGNED) {
          tmp = reinterpret_cast<const float4 *>(&B[row * n + bCol])[0];
        } else {
          tmp.x = (bCol + 0 < n) ? B[row * n + bCol + 0] : 0.0f;
          tmp.y = (bCol + 1 < n) ? B[row * n + bCol + 1] : 0.0f;
          tmp.z = (bCol + 2 < n) ? B[row * n + bCol + 2] : 0.0f;
          tmp.w = (bCol + 3 < n) ? B[row * n + bCol + 3] : 0.0f;
        }
      }
      reinterpret_cast<float4 *>(&shared_B[row * BLOCK_N + bCol])[0] = tmp;
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

  bool aligned = (k % 4 == 0) && (n % 4 == 0) &&
                 ((unsigned long long)A % 16 == 0) &&
                 ((unsigned long long)B % 16 == 0);

  if (args->stream) {
    if (aligned) {
      _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M, TILE_N, true>
          <<<gridDim, blockDim, 0, args->stream>>>(alpha, A, B, beta, C, m, k, n);
    } else {
      _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M, TILE_N, false>
          <<<gridDim, blockDim, 0, args->stream>>>(alpha, A, B, beta, C, m, k, n);
    }
  } else {
    if (aligned) {
      _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M, TILE_N, true>
          <<<gridDim, blockDim>>>(alpha, A, B, beta, C, m, k, n);
    } else {
      _cougar_kernel<BLOCK_M, BLOCK_K, BLOCK_N, TILE_M, TILE_N, false>
          <<<gridDim, blockDim>>>(alpha, A, B, beta, C, m, k, n);
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("_cougar_kernel launch error: %s\n", cudaGetErrorString(err));
}

void multiply_cougar(const float *A, const float *B, float *C, int m, int k,
                     int n, kernel_args_t *args) {
  multiply_cougar(ALPHA_DEFAULT, A, B, BETA_DEFAULT, C, m, k, n, args);
}
