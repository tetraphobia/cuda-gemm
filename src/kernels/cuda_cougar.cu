#include "cuda_cougar.h"
#include "types.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

#define THREADS_PER_ROW 8 // rows per thread
#define THREADS_PER_COL 8 // columns per thread

/**
 * Multiply two matrices `A` and `B` using CUDA.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 *
 */
__global__ void _cougar_shuffle_kernel(const float *A, const float *B, float *C,
                                       int m, int k, int n) {}

void multiply_cougar(const float *A, const float *B, float *C, int m, int k,
                     int n, kernel_args_t *args) {
  dim3 block(TILE_M / THREADS_PER_ROW, TILE_N / THREADS_PER_COL);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  if (args->stream)
    _cougar_shuffle_kernel<<<grid, block, 0, args->stream>>>(A, B, C, m, k, n);
  else
    _cougar_shuffle_kernel<<<grid, block>>>(A, B, C, m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("_cougar_shuffle_kernel launch error: %s\n",
           cudaGetErrorString(err));
}
