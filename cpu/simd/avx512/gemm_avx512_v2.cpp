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
) {
    // Initialize C to zero
    C.assign(M * N, 0.0);
    
    for (size_t i = 0; i < M; ++i) {

        size_t j = 0;

        // Process 8 doubles at a time
        for (; j + 7 < N; j += 8) {

            // Load 8 C values once
            __m512d c_vec = _mm512_loadu_pd(
                &C[i * N + j]
            );

            // Accumulate over K
            for (size_t k = 0; k < K; ++k) {

                // Broadcast one A value
                __m512d a_vec = _mm512_set1_pd(
                    A[i * K + k]
                );

                // Load 8  consecutive B values
                __m512d b_vec = _mm512_loadu_pd(
                    &B[k * N + j]
                );

                // c = a * b + c
                c_vec = _mm512_fmadd_pd(
                    a_vec,
                    b_vec,
                    c_vec
                );
            }

            // Store after completing all K accumulations
            _mm512_storeu_pd(
                &C[i * N + j],
                c_vec
            );
        }

        // Remaining columns
        for (; j < N; ++j) {

            for (size_t k = 0; k < K; ++k) {

                C[i * N + j] +=
                    A[i * K + k] *
                    B[k * N + j];
            }
        }
    }
}