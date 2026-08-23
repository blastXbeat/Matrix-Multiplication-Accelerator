#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
using namespace std;

void gemm_blocked(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M, size_t N, size_t K, size_t BLOCK_SIZE
);

int main(){
    size_t M = 4;
    size_t N = 4;
    size_t K = 4;
    size_t BLOCK_SIZE = 2;
    vector<double> A(M * K);
    vector<double> B(K * N);
    vector<double> C(M * N, 0.0);

    // Initialize matrices A and B with some values
    // A =
    // [ 1  2  3  4 ]
    // [ 2  3  4  5 ]
    // [ 3  4  5  6 ]
    // [ 4  5  6  7 ]
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            A[i * K + k] = i + k + 1; // Example initialization
        }
    }

    // B =
    // [ 1  2  3  4 ]
    // [ 2  3  4  5 ]
    // [ 3  4  5  6 ]
    // [ 4  5  6  7 ]   
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            B[k * N + j] = k + j + 1; // Example initialization
        }
    }

    // Call the blocked GEMM function
    gemm_blocked(A, B, C, M, N, K, BLOCK_SIZE);

    // Expected C = 
    // [ 30  40  50  60 ]
    // [ 40  54  68  82 ]
    // [ 50  68  86 104 ]
    // [ 60  82 104 126 ]

    vector<double> expected_C = {
    30, 40, 50, 60,
    40, 54, 68, 82,
    50, 68, 86, 104,
    60, 82, 104, 126
    };

    // Verify the result
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            cout << "C[" << i << "][" << j << "] = " << C[i * N + j] << ", expected = " << expected_C[i * N + j] << endl;
            assert(abs(C[i * N + j] - expected_C[i * N + j]) < 1e-9);
        }
    }

    cout << "Test passed!" << endl;
    return 0;
}