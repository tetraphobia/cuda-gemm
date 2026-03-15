#include "../src/kernels/cuda_cougar.h"
#include "../src/kernels/cuda_naive.h"
#include "../src/kernels/cuda_rf_block.h"
#include "../src/kernels/cuda_shared.h"
#include "../src/kernels/cuda_shuffle.h"
#include "../src/kernels/cuda_shuffle_nonshared.h"
#include "../src/kernels/cuda_shuffle_systolic.h"
#include "../src/kernels/types.h"

#include "../src/shared/matrix_utils.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <nvbench/nvbench.cuh>
#include <stdexcept>

#define M 4096
#define K 4096
#define N 4096

static void alloc_and_init(float **d_A, float **d_B, float **d_C) {
  float *h_A = (float *)malloc(sizeof(float) * M * K);
  float *h_B = (float *)malloc(sizeof(float) * K * N);

  init_matrix(h_A, M * K);
  init_matrix(h_B, K * N);

  cudaMalloc(d_A, sizeof(float) * M * K);
  cudaMalloc(d_B, sizeof(float) * K * N);
  cudaMalloc(d_C, sizeof(float) * M * N);

  cudaMemcpy(*d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(*d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemset(*d_C, 0, sizeof(float) * M * N);

  free(h_A);
  free(h_B);
}

static void free_matrices(float *d_A, float *d_B, float *d_C) {
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void naive(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_naive((const float *)A, (const float *)B, C, M, K, N, &args);
  });

  free_matrices(A, B, C);
}
// NVBENCH_BENCH(naive);

void shared(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_shared((const float *)A, (const float *)B, C, M, K, N, &args);
  });

  free_matrices(A, B, C);
}
// NVBENCH_BENCH(shared);

void rf_block(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_rf_block((const float *)A, (const float *)B, C, M, K, N,
                           &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(rf_block);

void warp_shuffle(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_warp_shuffle((const float *)A, (const float *)B, C, M, K, N,
                          &args);
  });

  free_matrices(A, B, C);
}
// NVBENCH_BENCH(warp_shuffle);

void cougar(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cougar((const float *)A, (const float *)B, C, M, K, N, &args);
  });

  free_matrices(A, B, C);
}
// NVBENCH_BENCH(cougar);

void shuffle_nonshared(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_warp_shuffle_nonshared((const float *)A, (const float *)B, C, M, K,
                                    N, &args);
  });

  free_matrices(A, B, C);
}
// NVBENCH_BENCH(shuffle_nonshared);
//
void shuffle_systolic(nvbench::state &state) {
  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_warp_shuffle_systolic((const float *)A, (const float *)B, C, M, K,
                                   N, &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(shuffle_systolic);

void cutlass_bench(nvbench::state &state) {
  float *A, *B, *C;
  alloc_and_init(&A, &B, &C);

  using RowMajor = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor,
                                           float, RowMajor, float>;
  Gemm gemm_op;

  float alpha = 1.0f;
  float beta = 0.0f;

  Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N},
                       {alpha, beta});

  // Warmup
  gemm_op(args, nullptr, nullptr);
  cudaDeviceSynchronize();

  state.exec([&](nvbench::launch &launch) {
    cutlass::Status status = gemm_op(args, nullptr, launch.get_stream());
    if (status != cutlass::Status::kSuccess)
      throw std::runtime_error("CUTLASS GEMM failed: " +
                               std::string(cutlassGetStatusString(status)));
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(cutlass_bench);

void cublas_bench(nvbench::state &state) {
  float *A, *B, *C;
  alloc_and_init(&A, &B, &C);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  // Warmup — forces internal workspace allocation before timing starts
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
  cudaDeviceSynchronize();

  state.exec([&](nvbench::launch &launch) {
    cublasSetStream(handle, launch.get_stream());
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
                &beta, C, N);
  });

  cublasDestroy(handle);
  free_matrices(A, B, C);
}
NVBENCH_BENCH(cublas_bench);
