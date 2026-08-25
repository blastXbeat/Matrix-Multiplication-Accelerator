#include <immintrin.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
using namespace std;

void gemm_avx512(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M,
    size_t N,
    size_t K    
);

int main(){
    size_t M = 18;
    size_t N = 18;
    size_t K = 18;

    vector<double> A(M * K);
    vector<double> B(K * N);
    vector<double> C(M * N);
    
    // Set values for A and B as [1, 2, 3, ..., M*K] and [1, 2, 3, ..., K*N]

    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<double>(i + 1);
    }

    for (size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<double>(i + 1);
    }

    // Call the gemm_avx2 function
    gemm_avx512(A, B, C, M, N, K);

    // Expect C to be a matrix where each element is the sum of products of corresponding elements from A and B
    // Assert that the result is as expected
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double expected_value = 0.0;
            for (size_t k = 0; k < K; ++k) {
                expected_value += A[i * K + k] * B[k * N + j];
            }
            assert(abs(C[i * N + j] - expected_value) < 1e-9);
        }
    }
    
    cout << "AVX2 GEMM test passed!" << endl;

    return 0;
}