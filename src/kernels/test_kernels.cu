#include "../shared/matrix_utils.h"
#include "cpu_naive.h"
#include "cuda_naive.cuh"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define M 3
#define K 2
#define N 3

int main(void) {
  float *A, *B, *C, *D;

  cudaMallocManaged(&A, sizeof(float) * M * K);
  cudaMallocManaged(&B, sizeof(float) * K * N);
  cudaMallocManaged(&C, sizeof(float) * M * N);
  cudaMallocManaged(&D, sizeof(float) * M * N);

  init_matrix(A, M * K);
  init_matrix(B, K * N);
  clear_matrix(C, M * N);
  clear_matrix(D, M * N);

  printf("A:\n");
  print_matrix(A, M, K);
  printf("\n");

  printf("B:\n");
  print_matrix(B, K, N);
  printf("\n");

  // Naive CPU test
  multiply_cpu_naive((const float *)A, (const float *)B, C, M, K, N);
  printf("TEST CPU NAIVE (BASE CASE)\n--------------\n");
  print_matrix(C, M, N);
  printf("\n");

  // Naive CUDA test
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);

  multiply_cuda_naive<<<grid, block>>>((const float *)A, (const float *)B, D, M,
                                       K, N);
  cudaDeviceSynchronize();

  printf("TEST CUDA NAIVE ");
  for (int i = 0; i < M * N; i++) {
    if (C[i] != D[i]) {
      printf("(FAIL!)\n");
      print_matrix(D, M, N);
      return 1;
    }
  }
  printf("(PASS!)\n");
  printf("--------------\n");

  print_matrix(D, M, N);

  return 0;
}
