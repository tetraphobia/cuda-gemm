#include "cpu_naive.h"

/**
 * Multiply two matrices `A` and `B` using CPU
 * and store result in matrix `C`.
 *
 * Assumes all matrices are 1D arrays with row-major ordering.
 *
 * Matrix `A` should have `m` rows and `k` columns.
 * Matrix `B` should have `k` rows and `n` columns.
 * Resulting matrix `C` should have `m` rows and `n` columns.
 */
template <typename T>
static void cpu_naive_impl(const T *A, const T *B, T *C, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C[i * n + j] = T(0);

      for (int l = 0; l < k; l++) {
        C[i * n + j] += A[i * k + l] * B[l * n + j];
      }
    }
  }
}

void multiply_cpu_naive(const float *A, const float *B, float *C, int m, int k, int n) {
  cpu_naive_impl(A, B, C, m, k, n);
}

void multiply_cpu_naive_double(const double *A, const double *B, double *C, int m, int k, int n) {
  cpu_naive_impl(A, B, C, m, k, n);
}
