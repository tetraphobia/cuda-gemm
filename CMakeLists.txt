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

add_executable(benchmark
  benchmarks/benchmark.cu
  src/kernels/cpu_naive.cpp
  src/kernels/cuda_naive.cu
  src/kernels/cuda_coalesced.cu
  src/kernels/cuda_rf_block.cu
  src/kernels/cuda_shared.cu
  src/kernels/cuda_tensor_core.cu
  src/shared/matrix_utils.cpp
)

add_executable(benchmark_lineinfo
  benchmarks/benchmark.cu
  src/kernels/cpu_naive.cpp
  src/kernels/cuda_naive.cu
  src/kernels/cuda_coalesced.cu
  src/kernels/cuda_rf_block.cu
  src/kernels/cuda_shared.cu
  src/kernels/cuda_tensor_core.cu
  src/shared/matrix_utils.cpp
)

add_executable(test
  test/test_kernels.cu
  src/kernels/cpu_naive.cpp
  src/kernels/cuda_naive.cu
  src/kernels/cuda_coalesced.cu
  src/kernels/cuda_rf_block.cu
  src/kernels/cuda_shared.cu
  src/kernels/cuda_tensor_core.cu
)

target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> # make cutlass shut up
)
set_target_properties(benchmark PROPERTIES CUDA_ARCHITECTURES "native")
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "native")

target_compile_options(benchmark_lineinfo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr> # make cutlass shut up AND compile with lineinfo for nsight
)
set_target_properties(benchmark_lineinfo PROPERTIES CUDA_ARCHITECTURES "native")

add_custom_target(lineinfo
    DEPENDS benchmark_lineinfo
    COMMENT "Build with -lineinfo for Nsight profiling"
)

target_link_libraries(benchmark
    nvbench::main
    nvidia::cutlass::cutlass
    CUDA::cublas
)

target_link_libraries(benchmark_lineinfo
    nvbench::main
    nvidia::cutlass::cutlass
    CUDA::cublas
)
