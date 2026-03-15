cmake_minimum_required(VERSION 3.30.4)
project(cuda-gemm LANGUAGES CXX CUDA)

include(cmake/CPM_0.42.0.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")
CPMAddPackage(
    NAME cutlass
    GITHUB_REPOSITORY NVIDIA/cutlass
    GIT_TAG main
    OPTIONS
    "CUTLASS_ENABLE_HEADERS_ONLY ON"
)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(benchmark
  benchmarks/benchmark.cu
  src/kernels/cpu_naive.cpp
  src/kernels/cuda_cougar.cu
  src/kernels/cuda_naive.cu
  src/kernels/cuda_rf_block.cu
  src/kernels/cuda_shared.cu
  src/kernels/cuda_shuffle.cu
  src/kernels/cuda_shuffle_systolic.cu
  src/kernels/cuda_shuffle_nonshared.cu
  src/kernels/cuda_shuffle_claude.cu
  src/shared/matrix_utils.cpp
)

add_executable(test
  test/test_kernels.cu
  src/kernels/cpu_naive.cpp
  src/kernels/cuda_cougar.cu
  src/kernels/cuda_naive.cu
  src/kernels/cuda_rf_block.cu
  src/kernels/cuda_shared.cu
  src/kernels/cuda_shuffle_nonshared.cu
  src/kernels/cuda_shuffle.cu
  src/kernels/cuda_shuffle_systolic.cu
  src/kernels/cuda_shuffle_claude.cu
)

target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> # make cutlass shut up
)

target_link_libraries(benchmark
    nvbench::main
    nvidia::cutlass::cutlass
    CUDA::cublas
)

