cmake_minimum_required(VERSION 3.30.4)
project(cuda-gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/nvbench)

add_executable(benchmark
  benchmarks/benchmark.cu
)

target_link_libraries(benchmark
  PRIVATE nvbench::nvbench
)
