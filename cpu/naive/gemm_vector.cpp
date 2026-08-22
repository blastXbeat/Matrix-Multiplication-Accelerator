#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

void gemm_vector(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C
)
{
    size_t m = A.size();
    size_t n = B[0].size();
    size_t p = B.size();

    assert(m == C.size());
    assert(n == C[0].size());
    assert(p == A[0].size());

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            C[i][j] = 0;
            for (size_t k = 0; k < p; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
