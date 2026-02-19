#include "../src/kernels/cpu_naive.h"
#include "../src/kernels/cuda_naive.h"
#include "../src/kernels/cuda_rf_block.h"
#include "../src/kernels/cuda_shared.h"
#include "../src/kernels/cuda_shuffle.h"
#include "../src/kernels/types.h"

#include "../src/shared/matrix_utils.h"
#include <cstdlib>
#include <cutlass/gemm/device/gemm.h>
#include <nvbench/nvbench.cuh>
#include <stdexcept>

#define M 1024
#define K 1024
#define N 1024

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
// NVBENCH_BENCH(cpu).set_is_cpu_only(true);

void naive(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_naive((const float *)A, (const float *)B, C, M, K, N, &args);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
// NVBENCH_BENCH(naive);

void shared(nvbench::state &state) {
  float *A, *B, *C;

  kernel_args_t args = KERNEL_ARGS_DEFAULT;

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_shared((const float *)A, (const float *)B, C, M, K, N, &args);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
// NVBENCH_BENCH(shared);

void rf_block(nvbench::state &state) {
  float *A, *B, *C;

  kernel_args_t args = KERNEL_ARGS_DEFAULT;

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_rf_block((const float *)A, (const float *)B, C, M, K, N,
                           &args);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
NVBENCH_BENCH(rf_block);

void warp_shuffle(nvbench::state &state) {
  float *A, *B, *C;

  kernel_args_t args = KERNEL_ARGS_DEFAULT;

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_warp_shuffle((const float *)A, (const float *)B, C, M, K, N,
                          &args);
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
// NVBENCH_BENCH(warp_shuffle);

void cutlass_bench(nvbench::state &state) {
  float *A, *B, *C;

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);

  using RowMajor = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor,
                                           float, RowMajor, float>;

  Gemm gemm_op;

  float alpha = 1.0f;
  float beta = 0.0f;

  Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N},
                       {alpha, beta});

  state.exec([&](nvbench::launch &launch) {
    cutlass::Status status = gemm_op(args, nullptr, launch.get_stream());

    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM failed: " +
                               std::string(cutlassGetStatusString(status)));
    }
  });

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
NVBENCH_BENCH(cutlass_bench);
