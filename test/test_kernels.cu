#include "../src/kernels/cuda_cougar.h"
#include "../src/kernels/cuda_naive.h"
#include "../src/kernels/cuda_coalesced.h"
#include "../src/kernels/cuda_rf_block.h"
#include "../src/kernels/cuda_shared.h"
#include "../src/kernels/cuda_shuffle.h"
#include "../src/kernels/cuda_shuffle_claude.h"
#include "../src/kernels/types.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Types
// ─────────────────────────────────────────────────────────────────────

typedef void (*kernel_fn_t)(const float *, const float *, float *, int, int,
                            int, kernel_args_t *);

typedef struct {
  const char *name;
  kernel_fn_t fn;
} kernel_entry_t;

typedef struct {
  // Absolute error stats
  float abs_max;    // max absolute error over all elements
  float abs_mean;   // mean absolute error
  float abs_stddev; // std dev of absolute errors

  // Relative error stats (skips reference elements near zero)
  float rel_max;    // max relative error
  float rel_mean;   // mean relative error (over non-tiny elements)
  float rel_stddev; // std dev of relative errors

  // RMSE
  float rmse;

  // Pass/fail
  int num_violations; // elements exceeding BOTH tolerances
  int total;
  int passed;
} error_stats_t;

typedef struct {
  int m, k, n;
  const char *desc;
} test_dim_t;

// Reference GEMM function.
void cpu_gemm(const float *A, const float *B, float *C, int m, int k, int n) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int p = 0; p < k; ++p)
        sum += (double)A[i * k + p] * (double)B[p * n + j];
      C[i * n + j] = (float)sum;
    }
}

// Elements whose absolute reference value is below this are excluded
// from relative error calculation to avoid division-by-near-zero noise.
#define REF_NEAR_ZERO 1e-4f

error_stats_t compute_error_stats(const float *ref, const float *got, int rows,
                                  int cols, float abs_tol, float rel_tol) {
  error_stats_t s = {0};
  s.total = rows * cols;

  float *abs_errs = (float *)malloc(s.total * sizeof(float));
  float *rel_errs = (float *)malloc(s.total * sizeof(float));
  int rel_count = 0;

  // Collect per-element errors
  double abs_sum = 0.0, rel_sum = 0.0, sq_sum = 0.0;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int idx = i * cols + j;
      float r = ref[idx];
      float g = got[idx];
      float abs_err = fabsf(r - g);

      abs_errs[idx] = abs_err;
      if (abs_err > s.abs_max)
        s.abs_max = abs_err;
      abs_sum += abs_err;
      sq_sum += (double)abs_err * abs_err;

      if (fabsf(r) > REF_NEAR_ZERO) {
        float rel_err = abs_err / fabsf(r);
        rel_errs[rel_count++] = rel_err;
        if (rel_err > s.rel_max)
          s.rel_max = rel_err;
        rel_sum += rel_err;
      }

      if (abs_err > abs_tol &&
          (fabsf(r) < REF_NEAR_ZERO || abs_err / fabsf(r) > rel_tol))
        s.num_violations++;
    }
  }

  s.abs_mean = (float)(abs_sum / s.total);
  s.rmse = sqrtf((float)(sq_sum / s.total));
  s.rel_mean = rel_count > 0 ? (float)(rel_sum / rel_count) : 0.0f;
  s.passed = (s.num_violations == 0);

  // Calculate standard deviation
  double abs_var = 0.0, rel_var = 0.0;
  rel_count = 0;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int idx = i * cols + j;
      float diff = abs_errs[idx] - s.abs_mean;
      abs_var += (double)diff * diff;

      float r = ref[idx];
      if (fabsf(r) > REF_NEAR_ZERO) {
        float rel_diff = rel_errs[rel_count++] - s.rel_mean;
        rel_var += (double)rel_diff * rel_diff;
      }
    }
  }

  s.abs_stddev = sqrtf((float)(abs_var / s.total));
  s.rel_stddev = rel_count > 1 ? sqrtf((float)(rel_var / rel_count)) : 0.0f;

  free(abs_errs);
  free(rel_errs);
  return s;
}

// Pretty print statistics
void print_error_stats(const char *kernel_name, int m, int k, int n,
                       const error_stats_t *s) {
  printf("  %-16s  %s\n", kernel_name, s->passed ? "PASS" : "FAIL");
  printf("    Absolute error:  max=%-12.4e  mean=%-12.4e  stddev=%.4e\n",
         s->abs_max, s->abs_mean, s->abs_stddev);
  printf("    Relative error:  max=%-12.4e  mean=%-12.4e  stddev=%.4e\n",
         s->rel_max, s->rel_mean, s->rel_stddev);
  printf("    RMSE: %.4e   violations: %d/%d\n", s->rmse, s->num_violations,
         s->total);
}

// Print a comparison table for all kernels on one test case
void print_comparison_table(kernel_entry_t *kernels, error_stats_t *stats,
                            int num_kernels) {
  printf("\n  %-16s  %-6s  %-12s  %-12s  %-12s  %-10s\n", "Kernel", "Result",
         "AbsMax", "RelMax", "RMSE", "Violations");
  printf("  %s\n", "-----------------------------------------------------------"
                   "-------------------------");
  for (int i = 0; i < num_kernels; ++i) {
    printf("  %-16s  %-6s  %-12.4e  %-12.4e  %-12.4e  %d/%d\n", kernels[i].name,
           stats[i].passed ? "PASS" : "FAIL", stats[i].abs_max,
           stats[i].rel_max, stats[i].rmse, stats[i].num_violations,
           stats[i].total);
  }
}

// Simple kernel runner
error_stats_t run_kernel(kernel_entry_t *kernel, int m, int k, int n,
                         const float *h_A, const float *h_B, const float *h_ref,
                         float abs_tol, float rel_tol, int print_mismatches) {
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, m * k * sizeof(float));
  cudaMalloc(&d_B, k * n * sizeof(float));
  cudaMalloc(&d_C, m * n * sizeof(float));
  cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, m * n * sizeof(float));

  kernel_args_t args = {0};
  kernel->fn(d_A, d_B, d_C, m, k, n, &args);

  error_stats_t s = {0};
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("  [%s] CUDA ERROR: %s\n", kernel->name, cudaGetErrorString(err));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return s;
  }

  float *h_got = (float *)malloc(m * n * sizeof(float));
  cudaMemcpy(h_got, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  s = compute_error_stats(h_ref, h_got, m, n, abs_tol, rel_tol);

  // Optionally print the first few mismatches
  if (print_mismatches && !s.passed) {
    int shown = 0;
    for (int i = 0; i < m && shown < 5; ++i) {
      for (int j = 0; j < n && shown < 5; ++j) {
        int idx = i * n + j;
        float abs_err = fabsf(h_ref[idx] - h_got[idx]);
        if (abs_err > abs_tol) {
          printf("    mismatch [%d,%d]: ref=%.6f  got=%.6f  abs_err=%.4e\n", i,
                 j, h_ref[idx], h_got[idx], abs_err);
          shown++;
        }
      }
    }
  }

  free(h_got);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return s;
}

// Random fill
static void rand_fill(float *buf, int n) {
  for (int i = 0; i < n; ++i)
    buf[i] = (float)(rand() % 200 - 100) / 20.0f; // [-5.0, 5.0]
}

// Test suite
static const test_dim_t TEST_DIMS[] = {
    // Exact tile multiples
    {128, 16, 128, "exact-tile"},
    {256, 32, 256, "exact-tile-large"},
    // Boundary conditions
    {127, 15, 127, "non-multiple"},
    {129, 17, 129, "just-over-tile"},
    {1, 16, 1, "single-row"},
    {13, 17, 11, "small-odd"},
    // Rectangular
    {256, 64, 128, "rect-tall"},
    {128, 64, 256, "rect-wide"},
    // Large K (stress-tests accumulation error)
    {64, 1024, 64, "large-k"},
    // Stress
    {512, 512, 512, "large-square"},
};

int main(void) {
  srand(42);

  // Tolerances
  const float ABS_TOL = 1e-2f; // looser for large-k cases
  const float REL_TOL = 1e-3f;

  kernel_entry_t kernels[] = {
      {"naive", multiply_cuda_naive},
      {"coalesced", multiply_cuda_coalesced},
      {"shared", multiply_cuda_shared},
      {"rf_block", multiply_cuda_rf_block},
      {"warp_shuffle", multiply_warp_shuffle},
      {"shuffle_claude", multiply_shuffle_claude},
      {"cougar", multiply_cougar},
  };
  int num_kernels = sizeof(kernels) / sizeof(kernels[0]);
  int num_dims = sizeof(TEST_DIMS) / sizeof(TEST_DIMS[0]);

  int total_passed = 0, total_run = 0;

  // Per-kernel aggregate stats across all test cases
  float per_kernel_max_abs[num_kernels];
  float per_kernel_max_rel[num_kernels];
  memset(per_kernel_max_abs, 0, sizeof(per_kernel_max_abs));
  memset(per_kernel_max_rel, 0, sizeof(per_kernel_max_abs));

  for (int di = 0; di < num_dims; ++di) {
    int m = TEST_DIMS[di].m;
    int k = TEST_DIMS[di].k;
    int n = TEST_DIMS[di].n;

    float *h_A = (float *)malloc(m * k * sizeof(float));
    float *h_B = (float *)malloc(k * n * sizeof(float));
    float *h_ref = (float *)malloc(m * n * sizeof(float));

    rand_fill(h_A, m * k);
    rand_fill(h_B, k * n);
    cpu_gemm(h_A, h_B, h_ref, m, k, n);

    printf("\n══ %s  (%d x %d x %d) ══\n", TEST_DIMS[di].desc, m, k, n);

    error_stats_t stats[num_kernels];
    for (int ki = 0; ki < num_kernels; ++ki) {
      stats[ki] =
          run_kernel(&kernels[ki], m, k, n, h_A, h_B, h_ref, ABS_TOL, REL_TOL,
                     /*print_mismatches=*/1);
      total_passed += stats[ki].passed;
      total_run++;

      if (stats[ki].abs_max > per_kernel_max_abs[ki])
        per_kernel_max_abs[ki] = stats[ki].abs_max;
      if (stats[ki].rel_max > per_kernel_max_rel[ki])
        per_kernel_max_rel[ki] = stats[ki].rel_max;
    }

    print_comparison_table(kernels, stats, num_kernels);

    free(h_A);
    free(h_B);
    free(h_ref);
  }

  // ── Summary ───────────────────────────────────────────────────────────────
  printf(
      "\n══════════════════════════════════════════════════════════════════\n");
  printf("SUMMARY  (%d/%d tests passed)\n\n", total_passed, total_run);
  printf("  %-16s  %-14s  %-14s\n", "Kernel", "WorstAbsMax", "WorstRelMax");
  printf("  %s\n", "------------------------------------------------");
  for (int ki = 0; ki < num_kernels; ++ki) {
    printf("  %-16s  %-14.4e  %-14.4e\n", kernels[ki].name,
           per_kernel_max_abs[ki], per_kernel_max_rel[ki]);
  }
  printf("\n");

  return (total_passed == total_run) ? 0 : 1;
}
