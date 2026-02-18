#ifndef CUDA_SHARED_H
#define CUDA_SHARED_H
#include "types.h"

void multiply_cuda_shared(const float *A, const float *B, float *C, int m,
                          int k, int n, kernel_args_t *args);

#endif
