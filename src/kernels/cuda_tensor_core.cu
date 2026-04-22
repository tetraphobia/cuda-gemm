#include "types.h"
#include <mma.h>
#include <stdio.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 32

#define WARPS_M 4
#define WARPS_N 2

__global__ void _tensor_core_kernel(const float *A, const float *B, float *C,
                                    int m, int k, int n) {
  using namespace nvcuda;
  // Shared memory for tiles
  __shared__ float smem_A[BLOCK_M][BLOCK_K]; // 128x32
  __shared__ float smem_B[BLOCK_K][BLOCK_N]; // 32x128

  // Staging buffer for C bounds checking
  // Each warp needs a 16x16 buffer. There are 8 warps total in the block.
  __shared__ float smem_C_warp[8][16][16];

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;
  int tid = warp_id * 32 + lane_id; // 0 to 255

  // 4 warps in M, 2 warps in N
  int warp_id_m = warp_id / WARPS_N; // 0 to 3
  int warp_id_n = warp_id % WARPS_N; // 0 to 1

  int warp_row_offset = warp_id_m * (BLOCK_M / WARPS_M); // 0, 32, 64, 96
  int warp_col_offset = warp_id_n * (BLOCK_N / WARPS_N); // 0, 64

  int global_row_offset = blockIdx.y * BLOCK_M;
  int global_col_offset = blockIdx.x * BLOCK_N;

  // Each warp computes 2x4 fragments of C
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[2][4];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      wmma::fill_fragment(frag_c[i][j], 0.0f);
    }
  }

  for (int kt = 0; kt < k; kt += BLOCK_K) {
    // 1. Load A tile into shared memory (128x32 = 4096 elements)
    // 256 threads -> 16 elements per thread
    for (int i = 0; i < 16; i++) {
      int idx = tid + i * 256;
      int r = idx / BLOCK_K;
      int c = idx % BLOCK_K;
      int global_r = global_row_offset + r;
      int global_c = kt + c;
      smem_A[r][c] = (global_r < m && global_c < k) ? A[global_r * k + global_c] : 0.0f;
    }

    // 2. Load B tile into shared memory (32x128 = 4096 elements)
    for (int i = 0; i < 16; i++) {
      int idx = tid + i * 256;
      int r = idx / BLOCK_N;
      int c = idx % BLOCK_N;
      int global_r = kt + r;
      int global_c = global_col_offset + c;
      smem_B[r][c] = (global_r < k && global_c < n) ? B[global_r * n + global_c] : 0.0f;
    }

    __syncthreads();

    // 3. MMA computation
    for (int sk = 0; sk < BLOCK_K; sk += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_a[2];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_b[4];

      // Load A fragments for this warp
      for (int i = 0; i < 2; i++) {
        wmma::load_matrix_sync(frag_a[i], &smem_A[warp_row_offset + i * WMMA_M][sk], BLOCK_K);
      }
      
      // Load B fragments for this warp
      for (int j = 0; j < 4; j++) {
        wmma::load_matrix_sync(frag_b[j], &smem_B[sk][warp_col_offset + j * WMMA_N], BLOCK_N);
      }

      // Multiply and accumulate
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
          wmma::mma_sync(frag_c[i][j], frag_a[i], frag_b[j], frag_c[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // 4. Store fragment C to global memory using warp staging
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      // Store one 16x16 fragment into staging buffer
      wmma::store_matrix_sync(&smem_C_warp[warp_id][0][0], frag_c[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      
      // 32 threads in the warp write 256 elements (8 per thread) to global memory
      for (int t = 0; t < 8; t++) {
        int idx = lane_id + t * 32;
        int r = idx / 16;
        int c = idx % 16;
        int global_r = global_row_offset + warp_row_offset + i * WMMA_M + r;
        int global_c = global_col_offset + warp_col_offset + j * WMMA_N + c;
        if (global_r < m && global_c < n) {
          C[global_r * n + global_c] = smem_C_warp[warp_id][r][c];
        }
      }
      __syncwarp();
    }
  }
}

void multiply_cuda_tensor_core(const float *A, const float *B, float *C, int m,
                               int k, int n, kernel_args_t *args) {
  dim3 block(32, WARPS_M * WARPS_N);
  int grid_y = (m + BLOCK_M - 1) / BLOCK_M;
  int grid_x = (n + BLOCK_N - 1) / BLOCK_N;
  dim3 grid(grid_x, grid_y);

  if (args->stream != NULL) {
    _tensor_core_kernel<<<grid, block, 0, args->stream>>>(A, B, C, m, k, n);
  } else {
    _tensor_core_kernel<<<grid, block, 0>>>(A, B, C, m, k, n);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
