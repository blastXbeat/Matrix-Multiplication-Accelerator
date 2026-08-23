#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

void gemm_blocked(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M, size_t N, size_t K, size_t BLOCK_SIZE
){
    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                size_t i_max = min(ii + BLOCK_SIZE, M);
                size_t j_max = min(jj + BLOCK_SIZE, N);
                size_t k_max = min(kk + BLOCK_SIZE, K);
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        double a= A[i * K + k];
                        for (size_t j = jj; j < j_max; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}