#include "../shared/matrix_utils.h"
#include "cpu_naive.h"
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 6
#define M 3
#define K 2
#define N 3

int main(void) {
  float *A = (float *)malloc(sizeof(float) * M * K);
  float *B = (float *)malloc(sizeof(float) * K * N);
  float *C = (float *)malloc(sizeof(float) * M * N);
  init_matrix(A, MATRIX_SIZE);
  init_matrix(B, MATRIX_SIZE);

  printf("TEST CPU NAIVE\n____________________\n");
  printf("A:\n");
  print_matrix(A, M, K);
  printf("\n");

  printf("B:\n");
  print_matrix(B, K, N);
  printf("\n");
  multiply_cpu_naive(A, B, C, M, K, N);

  printf("AB:\n");
  print_matrix(C, M, N);
  printf("\n");

  return 0;
}
