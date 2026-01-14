#include <stdio.h>
#include <stdlib.h>
#include "../../shared/matrix_utils.h"

#define MATRIX_SIZE 6
#define M 3
#define K 2
#define N 3

int main(void) {
    float * A = (float *) malloc(sizeof(float) * MATRIX_SIZE);
    float * B = (float *) malloc(sizeof(float) * MATRIX_SIZE);
    init_matrix(A, MATRIX_SIZE);
    init_matrix(B, MATRIX_SIZE);
    print_matrix(A, M, K);
    print_matrix(B, K, N);
    printf("Hello world\n");
}
