#ifndef COUGAR_H
#define COUGAR_H
#include "types.h"

void multiply_cougar(float alpha, const float *A, const float *B, float beta,
                     float *C, int m, int k, int n, kernel_args_t *args);
void multiply_cougar(const float *A, const float *B, float *C, int m, int k,
                     int n, kernel_args_t *args);

#endif
