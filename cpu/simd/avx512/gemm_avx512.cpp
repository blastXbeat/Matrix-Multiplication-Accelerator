#include <iostream>
#include <immintrin.h>
#include <vector>
using namespace std;

void gemm_avx512(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M,
    size_t N,
    size_t K
){
    C.assign(M * N, 0.0); // Initialize C to zero
    
    for (size_t i = 0; i < M; ++i){
        for (size_t k = 0; k < K; ++k){
            __m512d a_vec = _mm512_set1_pd(
                A[i*K + k]
            );
            
            size_t j = 0;
            for (; j + 7 < N ; j+= 8){

                __m512d b_vec = _mm512_loadu_pd(
                    &B[k*N + j]
                );

                __m512d c_vec = _mm512_loadu_pd(
                    &C[i*N + j]
                );

                c_vec = _mm512_fmadd_pd(
                    a_vec,
                    b_vec,
                    c_vec
                );

                _mm512_storeu_pd(
                    &C[i*N + j],
                    c_vec
                );
            }

            // Remaining elements
            for (;j < N; ++j){
                C[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}