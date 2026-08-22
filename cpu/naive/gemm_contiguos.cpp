#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

void gemm_contiguos(const vector<double>& A, const vector<double>& B, vector<double>& C, size_t M, size_t N, size_t K) {
    assert(A.size() == M * K);
    assert(B.size() == K * N);
    assert(C.size() == M * N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * N + j] = 0.0;
            for (size_t k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}