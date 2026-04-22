#include "types.h"
#include <stdio.h>

#define TILE_M 16
#define TILE_N 16

/**
 * Multiply two matrices `A` and `B` using CUDA
 * and store result in matrix `C`.
 *
 * Uses coalesced memory accesses for improved memory bandwidth utilization.
 */
template <typename T>
__global__ void _coalesced_kernel(const T *A, const T *B, T *C, int m,
                                  int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    T sum = T(0);

    for (int i = 0; i < k; i++) {
      sum += A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = sum;
  }
}

template <typename T>
static void multiply_cuda_coalesced_impl(const T *A, const T *B, T *C, int m, int k,
                                         int n, kernel_args_t *args) {
  dim3 block(TILE_M, TILE_N);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  if (args->stream != NULL) {
    _coalesced_kernel<<<grid, block, 0, args->stream>>>(A, B, C, m, k, n);
  } else {
    _coalesced_kernel<<<grid, block, 0>>>(A, B, C, m, k, n);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}

void multiply_cuda_coalesced(const float *A, const float *B, float *C, int m, int k,
                             int n, kernel_args_t *args) {
  multiply_cuda_coalesced_impl(A, B, C, m, k, n, args);
}

void multiply_cuda_coalesced_double(const double *A, const double *B, double *C, int m, int k,
                                    int n, kernel_args_t *args) {
  multiply_cuda_coalesced_impl(A, B, C, m, k, n, args);
}
