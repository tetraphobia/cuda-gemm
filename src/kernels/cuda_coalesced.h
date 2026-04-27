#ifndef CUDA_COALESCED_H
#include "types.h"
#define CUDA_COALESCED_H

void multiply_cuda_coalesced(const float *A, const float *B, float *C, int m, int k,
                             int n, kernel_args_t *args);
void multiply_cuda_coalesced_double(const double *A, const double *B, double *C, int m, int k,
                                    int n, kernel_args_t *args);

#endif
