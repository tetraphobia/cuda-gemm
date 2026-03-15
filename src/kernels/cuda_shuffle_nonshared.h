#ifndef WARP_SHUFFLE_NONSHARED_H
#define WARP_SHUFFLE_NONSHARED_H
#include "types.h"

void multiply_warp_shuffle_nonshared(const float *A, const float *B, float *C, int m,
                          int k, int n, kernel_args_t *args);

#endif
