#include <cblas.h>
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;

void gemm_openblas(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M,
    size_t N,
    size_t K
) {
    // Ensure that the input matrices are of the correct size
    if (A.size() != M * K || B.size() != K * N || C.size() != M * N) {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }

    // Perform the matrix multiplication using OpenBLAS
    cblas_dgemm(
        CblasRowMajor, // Row-major order
        CblasNoTrans,  // No transpose for A
        CblasNoTrans,  // No transpose for B
        M,             // Number of rows in A and C
        N,             // Number of columns in B and C
        K,             // Number of columns in A and rows in B
        1.0,           // Alpha (scaling factor for A*B)
        A.data(),      // Pointer to matrix A
        K,             // Leading dimension of A
        B.data(),      // Pointer to matrix B
        N,             // Leading dimension of B
        0.0,           // Beta (scaling factor for C)
        C.data(),      // Pointer to matrix C
        N              // Leading dimension of C
    );
}