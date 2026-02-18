#ifndef CUDA_RF_BLOCK_H
#define CUDA_RF_BLOCK_H
#include "types.h"

void multiply_cuda_rf_block(const float *A, const float *B, float *C, int m,
                            int k, int n, kernel_args_t *args);

#endif
