#ifndef WARP_SHUFFLE_SYSTOLIC_H
#define WARP_SHUFFLE_SYSTOLIC_H
#include "types.h"

void multiply_warp_shuffle_systolic(const float *A, const float *B, float *C, int m,
                          int k, int n, kernel_args_t *args);

#endif
