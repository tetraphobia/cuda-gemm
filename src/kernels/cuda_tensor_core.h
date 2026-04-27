#ifndef CUDA_TENSOR_CORE_H
#include "types.h"
#define CUDA_TENSOR_CORE_H

void multiply_cuda_tensor_core(const float *A, const float *B, float *C, int m, int k,
                         int n, kernel_args_t * args);

#endif
