#include "cuda_cougar.h"
#include "types.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

#define WARP_SIZE 32
#define ROW_WARPS 4 // warps along M dimension
#define COL_WARPS 4 // warps along N dimension
#define WARPS_PER_BLOCK (ROW_WARPS * COL_WARPS)
#define CHUNK_SIZE (TILE_M / ROW_WARPS)

/**
 * Multiply two matrices `A` and `B` using CUDA.
 * A modification of the shuffle_claude kernel.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 */
__global__ void _cougar_shuffle_kernel(const float *A, const float *B, float *C,
                                       int m, int k, int n) {
  const int lane = threadIdx.x;   // 0..31
  const int warpId = threadIdx.y; // 0..15

  // Decompose warpId into independent row and column warp indices
  const int rowWarp = warpId / COL_WARPS; // 0..3 — which row chunk
  const int colWarp = warpId % COL_WARPS; // 0..3 — which col slice

  const int blockRow = blockIdx.y * TILE_M;
  const int blockCol = blockIdx.x * TILE_N;

  const int rowChunkBase = rowWarp * CHUNK_SIZE; // 0, 32, 64, 96
  const int warpColBase =
      blockCol + colWarp * WARP_SIZE; // independent of rowWarp

  __shared__ float As[TILE_M][TILE_K];

  float acc[CHUNK_SIZE];

  for (int i = 0; i < CHUNK_SIZE; ++i)
    acc[i] = 0.0f;

  const unsigned int FULL_MASK = 0xffffffffu;

  for (int kt = 0; kt < k; kt += TILE_K) {

    {
      int tid = warpId * WARP_SIZE + lane;
      int total = TILE_M * TILE_K;
      for (int idx = tid; idx < total; idx += WARPS_PER_BLOCK * WARP_SIZE) {
        int r = idx / TILE_K;
        int c = idx % TILE_K;
        int gr = blockRow + r;
        int gc = kt + c;
        As[r][c] = (gr < m && gc < k) ? A[gr * k + gc] : 0.0f;
      }
    }

    __syncthreads();

    for (int kk = 0; kk < TILE_K; ++kk) {
      int bCol = warpColBase + lane;
      int bRow = kt + kk;
      float myB = (bRow < k && bCol < n) ? B[bRow * n + bCol] : 0.0f;

      float b = __shfl_sync(FULL_MASK, myB, lane);

      for (int i = 0; i < CHUNK_SIZE; ++i)
        acc[i] += As[rowChunkBase + i][kk] * b;
    }

    __syncthreads();
  }

  int outCol = warpColBase + lane;
  if (outCol < n) {
    for (int i = 0; i < CHUNK_SIZE; ++i) {
      int outRow = blockRow + rowChunkBase + i;
      if (outRow < m)
        C[outRow * n + outCol] = acc[i];
    }
  }
}

void multiply_cougar(const float *A, const float *B, float *C, int m, int k,
                     int n, kernel_args_t *args) {
  dim3 block(WARP_SIZE, WARPS_PER_BLOCK); // 32x16 = 512 threads
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
