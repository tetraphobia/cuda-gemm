#include "../src/kernels/cpu_naive.h"
#include "../src/kernels/cuda_coalesced.h"
#include "../src/kernels/cuda_naive.h"
#include "../src/kernels/cuda_rf_block.h"
#include "../src/kernels/cuda_shared.h"
#include "../src/kernels/cuda_shared_32.h"
#include "../src/kernels/cuda_tensor_core.h"
#include "../src/kernels/types.h"

#include "../src/shared/matrix_utils.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <nvbench/nvbench.cuh>
#include <stdexcept>

static void alloc_and_init(float **d_A, float **d_B, float **d_C, size_t M,
                           size_t K, size_t N) {
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

void cpu_naive_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *h_A = (float *)malloc(sizeof(float) * M * K);
  float *h_B = (float *)malloc(sizeof(float) * K * N);
  float *h_C = (float *)malloc(sizeof(float) * M * N);

  init_matrix(h_A, M * K);
  init_matrix(h_B, K * N);
  memset(h_C, 0, sizeof(float) * M * N);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    multiply_cpu_naive(h_A, h_B, h_C, M, K, N);
  });

  free(h_A);
  free(h_B);
  free(h_C);
}
// NVBENCH_BENCH(cpu_naive_bench).add_int64_axis("N", {64, 128, 256, 512,
// 1024});

void naive(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_naive((const float *)A, (const float *)B, C, M, K, N, &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(naive).add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096,
                                          8192, 16384});

void coalesced(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_coalesced((const float *)A, (const float *)B, C, M, K, N,
                            &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(coalesced).add_int64_axis("N", {64, 128, 256, 512, 1024, 2048,
                                              4096, 8192, 16384});

void shared(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_shared((const float *)A, (const float *)B, C, M, K, N, &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(shared).add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096,
                                           8192, 16384});

void shared_32(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_shared_32((const float *)A, (const float *)B, C, M, K, N,
                            &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(shared_32).add_int64_axis("N", {64, 128, 256, 512, 1024, 2048,
                                              4096, 8192, 16384});

void rf_block(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_rf_block((const float *)A, (const float *)B, C, M, K, N,
                           &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(rf_block).add_int64_axis("N", {64, 128, 256, 512, 1024, 2048,
                                             4096, 8192, 16384});

void tensor_core(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_tensor_core((const float *)A, (const float *)B, C, M, K, N,
                              &args);
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(tensor_core)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});


void cutlass_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  alloc_and_init(&A, &B, &C, M, K, N);

  using RowMajor = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor,
                                           float, RowMajor, float>;
  Gemm gemm_op;

  float alpha = 1.0f;
  float beta = 0.0f;

  Gemm::Arguments args(
      {static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)},
      {A, static_cast<int>(K)}, {B, static_cast<int>(N)},
      {C, static_cast<int>(N)}, {C, static_cast<int>(N)}, {alpha, beta});

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
NVBENCH_BENCH(cutlass_bench)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void cublas_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  alloc_and_init(&A, &B, &C, M, K, N);

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
NVBENCH_BENCH(cublas_bench)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void cutlass_tensor_core_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  alloc_and_init(&A, &B, &C, M, K, N);

  using RowMajor = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<
      float, RowMajor,                         // A
      float, RowMajor,                         // B
      float, RowMajor,                         // C
      float,                                   // accumulator
      cutlass::arch::OpClassTensorOp,          // tensor cores
      cutlass::arch::Sm80,                     // Ampere
      cutlass::gemm::GemmShape<128, 128, 16>,  // threadblock tile
      cutlass::gemm::GemmShape<64, 64, 16>,    // warp tile
      cutlass::gemm::GemmShape<16, 8, 8>       // TF32 MMA instruction
      >;
  Gemm gemm_op;

  float alpha = 1.0f;
  float beta = 0.0f;

  Gemm::Arguments args(
      {static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)},
      {A, static_cast<int>(K)}, {B, static_cast<int>(N)},
      {C, static_cast<int>(N)}, {C, static_cast<int>(N)}, {alpha, beta});

  // Warmup
  gemm_op(args, nullptr, nullptr);
  cudaDeviceSynchronize();

  state.exec([&](nvbench::launch &launch) {
    cutlass::Status status = gemm_op(args, nullptr, launch.get_stream());
    if (status != cutlass::Status::kSuccess)
      throw std::runtime_error("CUTLASS Tensor Core GEMM failed: " +
                               std::string(cutlassGetStatusString(status)));
  });

  free_matrices(A, B, C);
}
NVBENCH_BENCH(cutlass_tensor_core_bench)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void cublas_tensor_core_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  float *A, *B, *C;
  alloc_and_init(&A, &B, &C, M, K, N);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Route FP32 GEMMs through TF32 tensor cores (Ampere+).
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

  float alpha = 1.0f;
  float beta = 0.0f;

  // Warmup — done after SetMathMode so workspace matches the timed path.
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
NVBENCH_BENCH(cublas_tensor_core_bench)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

// ═══════════════════════════════════════════════════════════════════════════
// DOUBLE PRECISION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

static void alloc_and_init_double(double **d_A, double **d_B, double **d_C,
                                  size_t M, size_t K, size_t N) {
  double *h_A = (double *)malloc(sizeof(double) * M * K);
  double *h_B = (double *)malloc(sizeof(double) * K * N);

  init_matrix_double(h_A, M * K);
  init_matrix_double(h_B, K * N);

  cudaMalloc(d_A, sizeof(double) * M * K);
  cudaMalloc(d_B, sizeof(double) * K * N);
  cudaMalloc(d_C, sizeof(double) * M * N);

  cudaMemcpy(*d_A, h_A, sizeof(double) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(*d_B, h_B, sizeof(double) * K * N, cudaMemcpyHostToDevice);
  cudaMemset(*d_C, 0, sizeof(double) * M * N);

  free(h_A);
  free(h_B);
}

static void free_matrices_double(double *d_A, double *d_B, double *d_C) {
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void naive_double(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  double *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init_double(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_naive_double((const double *)A, (const double *)B, C, M, K, N,
                               &args);
  });

  free_matrices_double(A, B, C);
}
NVBENCH_BENCH(naive_double)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void coalesced_double(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  double *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init_double(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_coalesced_double((const double *)A, (const double *)B, C, M,
                                   K, N, &args);
  });

  free_matrices_double(A, B, C);
}
NVBENCH_BENCH(coalesced_double)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void shared_double(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  double *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init_double(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_shared_double((const double *)A, (const double *)B, C, M, K,
                                N, &args);
  });

  free_matrices_double(A, B, C);
}
NVBENCH_BENCH(shared_double)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void shared_32_double(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  double *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init_double(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_shared_32_double((const double *)A, (const double *)B, C, M,
                                   K, N, &args);
  });

  free_matrices_double(A, B, C);
}
NVBENCH_BENCH(shared_32_double)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void rf_block_double(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  double *A, *B, *C;
  kernel_args_t args = KERNEL_ARGS_DEFAULT;
  alloc_and_init_double(&A, &B, &C, M, K, N);

  state.exec([&](nvbench::launch &launch) {
    args.stream = launch.get_stream();
    multiply_cuda_rf_block_double((const double *)A, (const double *)B, C, M, K,
                                  N, &args);
  });

  free_matrices_double(A, B, C);
}
NVBENCH_BENCH(rf_block_double)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

void cublas_double_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto M = N;
  const auto K = N;

  double *A, *B, *C;
  alloc_and_init_double(&A, &B, &C, M, K, N);

  cublasHandle_t handle;
  cublasCreate(&handle);

  double alpha = 1.0;
  double beta = 0.0;

  // Warmup
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
  cudaDeviceSynchronize();

  state.exec([&](nvbench::launch &launch) {
    cublasSetStream(handle, launch.get_stream());
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
                &beta, C, N);
  });

  cublasDestroy(handle);
  free_matrices_double(A, B, C);
}
NVBENCH_BENCH(cublas_double_bench)
    .add_int64_axis("N", {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});
