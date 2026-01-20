#ifndef CUDA_RF_BLOCK_H
#define CUDA_RF_BLOCK_H

__global__ void multiply_cuda_rf_block(const float *A, const float *B, float *C,
                                       int m, int k, int n);

#endif
