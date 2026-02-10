#include "../src/kernels/cpu_naive.h"
#include "../src/kernels/cuda_naive.cuh"
#include "../src/kernels/cuda_rf_block.cuh"
#include "../src/kernels/cuda_shared.cuh"

#include "../src/shared/matrix_utils.h"
#include <cstdlib>
#include <nvbench/nvbench.cuh>

#define M 256
#define K 256
#define N 256

void cpu(nvbench::state &state) {
  float *A, *B, *C;

  A = (float *)malloc(sizeof(float) * M * K);
  B = (float *)malloc(sizeof(float) * M * K);
  C = (float *)malloc(sizeof(float) * M * K);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    multiply_cpu_naive((const float *)A, (const float *)B, C, M, K, N);
  });

  free(A);
  free(B);
  free(C);
}
NVBENCH_BENCH(cpu)
    .set_is_cpu_only(true);

void naive_16x16(nvbench::state &state) {
  float *A, *B, *C;
  int tile_size = 16;

  dim3 block(tile_size, tile_size);
  dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    multiply_cuda_naive<<<block, grid, 0, launch.get_stream()>>>(
        (const float *)A, (const float *)B, C, M, K, N);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
NVBENCH_BENCH(naive_16x16);

void naive_32x32(nvbench::state &state) {
  float *A, *B, *C;
  int tile_size = 32;

  dim3 block(tile_size, tile_size);
  dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    multiply_cuda_naive<<<block, grid, 0, launch.get_stream()>>>(
        (const float *)A, (const float *)B, C, M, K, N);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
NVBENCH_BENCH(naive_32x32);

void rf_block(nvbench::state &state) {
  float *A, *B, *C;
  int tile_size = 16;

  dim3 block(tile_size, tile_size);
  dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    multiply_cuda_rf_block<<<block, grid, 0, launch.get_stream()>>>(
        (const float *)A, (const float *)B, C, M, K, N);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
NVBENCH_BENCH(rf_block);

void shared_16x16(nvbench::state &state) {
  float *A, *B, *C;
  int tile_size = 16;

  dim3 block(tile_size, tile_size);
  dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    multiply_cuda_shared<<<block, grid, 0, launch.get_stream()>>>(
        (const float *)A, (const float *)B, C, M, K, N);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
NVBENCH_BENCH(shared_16x16);

void shared_32x32(nvbench::state &state) {
  float *A, *B, *C;
  int tile_size = 32;

  dim3 block(tile_size, tile_size);
  dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    multiply_cuda_shared<<<block, grid, 0, launch.get_stream()>>>(
        (const float *)A, (const float *)B, C, M, K, N);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
NVBENCH_BENCH(shared_32x32);

// void warp_shuffle(nvbench::state &state) {
//   float *A, *B, *C;
//
//   dim3 block(tile_size, tile_size);
//   dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) /
//   tile_size);
//
//   cudaMallocManaged(&A, sizeof(float) * M * K);
//   cudaMallocManaged(&B, sizeof(float) * K * N);
//   cudaMallocManaged(&C, sizeof(float) * M * N);
//
//   init_matrix(A, M * K);
//   init_matrix(B, K * N);
//   clear_matrix(C, M * N);
//
//   state.exec([&](nvbench::launch &launch) {
//     multiply_cuda_warp_shuffle<<<block, grid, 0, launch.get_stream()>>>(
//         (const float *)A, (const float *)B, C, M, K, N);
//   });
//
//   cudaFree(A);
//   cudaFree(B);
//   cudaFree(C);
// }
