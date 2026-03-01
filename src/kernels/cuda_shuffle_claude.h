#ifndef SHUFFLE_CLAUDE_H 
#define SHUFFLE_CLAUDE_H
#include "types.h"

void multiply_shuffle_claude(const float *A, const float *B, float *C, int m,
                          int k, int n, kernel_args_t *args);

#endif
