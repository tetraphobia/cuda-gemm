#ifndef CUDA_WARP_SHUFFLE_H
#define CUDA_WARP_SHUFFLE_H

__global__ void multiply_cuda_warp_shuffle(float *A, float *B, float *C, int m,
                                           int k, int n);

#endif
