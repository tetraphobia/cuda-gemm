#ifndef CUDA_NAIVE_H
#include "types.h"
#define CUDA_NAIVE_H

void multiply_cuda_naive(const float *A, const float *B, float *C, int m, int k,
                         int n, kernel_args_t * args);

#endif
