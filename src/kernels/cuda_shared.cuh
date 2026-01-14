#ifndef CUDA_SHARED_H
#define CUDA_SHARED_H

__global__ void multiply_cuda_shared(float *A, float *B, float *C, int m, int k,
                                     int n);

#endif
