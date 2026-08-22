#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

// Function Decelaration
void gemm_vector(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C
);

void gemm_contiguos(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M, size_t N, size_t K
);

int main()
{
    // A = [1 2]
    //     [3 4]
    vector<double> A_contig = {1, 2, 3, 4}; // 2x2 matrix

    // B = [5 6]
    //     [7 8]
    vector<double> B_contig = {5, 6, 7, 8}; // 2x2 matrix
    
    // Expected
    // C = [19 22]
    //     [43 50]
    vector<double> C_contig(4); // 2x2 matrix
    
    gemm_contiguos(A_contig, B_contig, C_contig, 2, 2, 2);

    assert(C_contig[0] == 19);
    assert(C_contig[1] == 22);
    assert(C_contig[2] == 43);
    assert(C_contig[3] == 50);

    cout << "All tests passed!" << endl;
    return 0;
}