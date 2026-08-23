#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <omp.h>
using namespace std;

void gemm_omp(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M,
    size_t N,
    size_t K,
    size_t BLOCK_SIZE
){
    assert(A.size() == M * K);
    assert(B.size() == K * N);
    assert(C.size() == M * N);

    #pragma omp parallel for schedule(static)
    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE){
        size_t i_max = min(ii + BLOCK_SIZE, M);

        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE){
            size_t k_max = min(kk + BLOCK_SIZE, K);
            
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE){
                size_t j_max = min(jj + BLOCK_SIZE, N);


                for (size_t i = ii; i < i_max; ++i){
                    for (size_t k = kk; k < k_max; ++k){

                        double a_ik = A[i * K + k];
                        for (size_t j = jj; j < j_max; ++j){
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}