#include "types.h"
#include <stdio.h>
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define WARPS_PER_BLOCK 4

__global__ void _tensor_core_kernel(const float *A, const float *B, float *C, int m, int k, int n) {
  using namespace nvcuda;
  // Shared memory for tiles
  // A tile: 64x8 (WARPS_PER_BLOCK * WMMA_M rows, WMMA_K cols)
  __shared__ float smem_A[WARPS_PER_BLOCK * WMMA_M][WMMA_K];
  // B tile: 8x16 (WMMA_K rows, WMMA_N cols)
  __shared__ float smem_B[WMMA_K][WMMA_N];
  // C tile: 64x16 (WARPS_PER_BLOCK * WMMA_M rows, WMMA_N cols)
  __shared__ float smem_C[WARPS_PER_BLOCK * WMMA_M][WMMA_N];

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;
  int tid = warp_id * 32 + lane_id;

  int global_row_offset = blockIdx.y * (WARPS_PER_BLOCK * WMMA_M);
  int global_col_offset = blockIdx.x * WMMA_N;

  // Initialize accumulator fragment
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
  wmma::fill_fragment(frag_c, 0.0f);

  int num_tiles = (k + WMMA_K - 1) / WMMA_K;

  for (int t = 0; t < num_tiles; t++) {
    // 1. Load A tile into shared memory (64x8 = 512 elements)
    // 128 threads -> 4 elements per thread
    for (int i = 0; i < 4; i++) {
      int idx = tid + i * 128;
      int r = idx / WMMA_K;
      int c = idx % WMMA_K;
      int global_r = global_row_offset + r;
      int global_c = t * WMMA_K + c;
      smem_A[r][c] = (global_r < m && global_c < k) ? A[global_r * k + global_c] : 0.0f;
    }

    // 2. Load B tile into shared memory (8x16 = 128 elements)
    // 128 threads -> 1 element per thread
    int r = tid / WMMA_N;
    int c = tid % WMMA_N;
    int global_r = t * WMMA_K + r;
    int global_c = global_col_offset + c;
    smem_B[r][c] = (global_r < k && global_c < n) ? B[global_r * n + global_c] : 0.0f;

    __syncthreads();

    // 3. Load from shared memory to fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_b;

    // Each warp loads its own 16x8 slice of A, but all warps share the same 8x16 slice of B
    wmma::load_matrix_sync(frag_a, &smem_A[warp_id * WMMA_M][0], WMMA_K);
    wmma::load_matrix_sync(frag_b, &smem_B[0][0], WMMA_N);

    // 4. MMA computation
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    __syncthreads();
  }

  // 5. Store fragment C to shared memory
  wmma::store_matrix_sync(&smem_C[warp_id * WMMA_M][0], frag_c, WMMA_N, wmma::mem_row_major);

  __syncthreads();

  // 6. Write shared memory C to global memory with bounds checking
  // smem_C is 64x16 = 1024 elements. 128 threads -> 8 elements per thread.
  for (int i = 0; i < 8; i++) {
    int idx = tid + i * 128;
    int r = idx / WMMA_N;
    int c = idx % WMMA_N;
    int global_r = global_row_offset + r;
    int global_c = global_col_offset + c;
    if (global_r < m && global_c < n) {
      C[global_r * n + global_c] = smem_C[r][c];
    }
  }
}

void multiply_cuda_tensor_core(const float *A, const float *B, float *C, int m, int k, int n, kernel_args_t *args) {
  dim3 block(32, WARPS_PER_BLOCK);
  int grid_y = (m + (WARPS_PER_BLOCK * WMMA_M) - 1) / (WARPS_PER_BLOCK * WMMA_M);
  int grid_x = (n + WMMA_N - 1) / WMMA_N;
  dim3 grid(grid_x, grid_y);

  if (args->stream != NULL) {
    _tensor_core_kernel<<<grid, block, 0, args->stream>>>(A, B, C, m, k, n);
  } else {
    _tensor_core_kernel<<<grid, block, 0>>>(A, B, C, m, k, n);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Tensor Core kernel launch error: %s\n", cudaGetErrorString(err));
}
