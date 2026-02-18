cmake_minimum_required(VERSION 3.30.4)
project(cuda-gemm LANGUAGES CXX CUDA)

include(cmake/CPM_0.42.0.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

add_executable(benchmark
  benchmarks/benchmark.cu
  src/kernels/cpu_naive.cpp
  src/kernels/cuda_naive.cu
  src/kernels/cuda_rf_block.cu
  src/kernels/cuda_shared.cu
  src/shared/matrix_utils.cpp
)

target_link_libraries(benchmark nvbench::main)
