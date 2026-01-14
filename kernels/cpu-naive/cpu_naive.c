#include "cpu_naive.h"

/**
 * Multiply two matrices `A` and `B` using CPU only
 * and store result in matrix `C`.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 * Matrix `A` should have `m` rows and `k` columns.
 * Matrix `B` should have `k` rows and `n` columns.
 * Resulting matrix `C` should have `m` rows and `n` columns.
 */
void multiply_cpu_naive(float *A, float *B, float *C, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C[i * n + j] = 0;

      for (int l = 0; l < k; l++) {
        C[i * n + j] += A[i * k + l] * B[l * n + j];
      }
    }
  }
}

