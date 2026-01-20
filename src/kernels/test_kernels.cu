#include "../shared/matrix_utils.h"
#include "cpu_naive.h"
#include "cuda_naive.cuh"
#include "cuda_rf_block.cuh"
#include "cuda_shared.cuh"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define M 3
#define K 2
#define N 3
#define TILE_SIZE 16

int main(void) {
  float *A, *B, *EXPECTED, *ACTUAL;

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&EXPECTED, sizeof(float) * M * N);
  cudaMallocManaged(&ACTUAL, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(EXPECTED, M * N);
  clear_matrix(ACTUAL, M * N);

  printf("A:\n");
  print_matrix(A, M, K);
  printf("\n");

  printf("B:\n");
  print_matrix(B, K, N);
  printf("\n");

  /*
   * CPU
   */

  // Naive CPU test
  multiply_cpu_naive((const float *)A, (const float *)B, EXPECTED, M, K, N);
  printf("TEST CPU NAIVE (BASE CASE)\n--------------\n");
  print_matrix(EXPECTED, M, N);
  printf("\n");

  /*
   * CUDA
   */

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  // Naive CUDA test
  multiply_cuda_naive<<<grid, block>>>((const float *)A, (const float *)B,
                                       ACTUAL, M, K, N);

  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  printf("TEST CUDA NAIVE ");
  for (int i = 0; i < M * N; i++) {
    if (EXPECTED[i] != ACTUAL[i]) {
      printf("(FAIL!)\n");
      print_matrix(ACTUAL, M, N);
      return 1;
    }
  }
  printf("(PASS!)\n");
  printf("--------------\n");

  print_matrix(ACTUAL, M, N);
  clear_matrix(ACTUAL, M * N);

  // SRAM CUDA test
  multiply_cuda_shared<<<grid, block>>>((const float *)A, (const float *)B,
                                        ACTUAL, M, K, N);

  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  printf("TEST CUDA SHARED ");
  for (int i = 0; i < M * N; i++) {
    if (EXPECTED[i] != ACTUAL[i]) {
      printf("(FAIL!)\n");
      print_matrix(ACTUAL, M, N);
      return 1;
    }
  }
  printf("(PASS!)\n");
  printf("--------------\n");

  print_matrix(ACTUAL, M, N);
  clear_matrix(ACTUAL, M * N);

  // SRAM + RF CUDA Test
  multiply_cuda_rf_block<<<grid, block>>>((const float *)A, (const float *)B,
                                          ACTUAL, M, K, N);

  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  printf("TEST CUDA RF BLOCK ");
  for (int i = 0; i < M * N; i++) {
    if (EXPECTED[i] != ACTUAL[i]) {
      printf("(FAIL!)\n");
      print_matrix(ACTUAL, M, N);
      return 1;
    }
  }
  printf("(PASS!)\n");
  printf("--------------\n");

  print_matrix(ACTUAL, M, N);
  clear_matrix(ACTUAL, M * N);

  return 0;
}
