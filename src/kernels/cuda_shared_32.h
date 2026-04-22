#ifndef CUDA_SHARED_32_H
#include "types.h"
#define CUDA_SHARED_32_H

void multiply_cuda_shared_32(const float *A, const float *B, float *C, int m, int k,
                             int n, kernel_args_t *args);

#endif
