#ifndef CUDA_NAIVE_H
#define CUDA_NAIVE_H

__global__ void multiply_cuda_naive(const float *A, const float *B, float *C, int m, int k,
                                    int n);

#endif
