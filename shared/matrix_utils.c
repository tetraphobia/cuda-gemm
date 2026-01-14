#include "matrix_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_matrix(float *A, int size) {
  srandom(time(NULL));

  for (int i = 0; i < size; i++) {
    A[i] = rand() % 20; // Will be skewed but doesn't matter.
  }
}

void print_matrix(float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%6.1f ", A[i * n + j]);
    }
    printf("\n");
  }
}

int main(void) {
    // Test these functions
    float * A = (float *) malloc(sizeof(float) * 6);

    init_matrix(A, 6);
    print_matrix(A, 3, 2);
}
