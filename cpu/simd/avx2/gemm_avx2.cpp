#include <iostream>
#include <immintrin.h>
#include <vector>
using namespace std;

void gemm_avx2(
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
            __m256d a_vec = _mm256_set1_pd(
                A[i*K + k]
            );
            
            size_t j = 0;
            for (; j + 3 < N ; j+= 4){

                __m256d b_vec = _mm256_loadu_pd(
                    &B[k*N + j]
                );

                __m256d c_vec = _mm256_loadu_pd(
                    &C[i*N + j]
                );

                c_vec = _mm256_fmadd_pd(
                    a_vec,
                    b_vec,
                    c_vec
                );

                _mm256_storeu_pd(
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