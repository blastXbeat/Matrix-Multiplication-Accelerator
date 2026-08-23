#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
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
);

int main(){
    const size_t M = 512;
    const size_t N = 512;
    const size_t K = 512;

    const size_t BLOCK_SIZE = 64;

    vector<double> A(M * K);
    vector<double> B(K * N);
    vector<double> C(M * N, 0.0);

    for (size_t i = 0; i < A.size(); i++) {
        A[i] = 1.0;
    }

    for (size_t i = 0; i < B.size(); i++) {
        B[i] = 2.0;
    }

    gemm_omp(A, B, C, M, N, K, BLOCK_SIZE);

    // Expecting C to be filled with 2.0 * K since A is filled with 1.0 and B is filled with 2.0.

    for (size_t i = 0; i < M * N; i++) {
        assert(abs(C[i] - 2.0 * K) < 1e-9);
    }

    cout << "OpenMP GEMM test passed!" << endl;

    cout << "Threads used: "
         << omp_get_max_threads()
         << endl;

    return 0;
}