#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__global__ void test_wmma_tf32(float *a, float *b, float *c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);
    wmma::load_matrix_sync(frag_a, a, 8);
    wmma::load_matrix_sync(frag_b, b, 16);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    wmma::store_matrix_sync(c, frag_c, 16, wmma::mem_row_major);
}
int main() { return 0; }
