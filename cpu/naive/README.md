# CPU Naive GEMM (General Matrix-Matrix Multiplication)

A baseline CPU implementation of General Matrix-Matrix Multiplication (GEMM) in C++ to evaluate algorithmic correctness and the impact of memory layout on cache behavior.

---

## Overview

General Matrix-Matrix Multiplication computes:

$$C = A \times B$$

Where:
* $A$ is an $M \times K$ matrix
* $B$ is a $K \times N$ matrix
* $C$ is an $M \times N$ matrix

Each element $C_{ij}$ is defined as:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$$

Both implementations execute the standard naive three-loop algorithm with $\mathcal{O}(M \times N \times K)$ computational complexity.

---

## Memory Layout Implementations

### 1. Vector of Vectors (`std::vector<std::vector<double>>`)

Represents the matrix as a top-level vector containing individual row vectors.

```cpp
std::vector<std::vector<double>> A = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0},
    {7.0, 8.0, 9.0}
};

// Access element at (i, j)
double val = A[i][j];

// Infer dimensions directly
size_t M = A.size();
size_t K = A[0].size();
```

* **Pros:** Intuitive 2D indexing (`A[i][j]`), self-describing dimensions.
* **Cons:** Pointer indirection overhead, scattered heap allocations, poor spatial locality, not SIMD-friendly.

---

### 2. Contiguous 1D Vector (`std::vector<double>`)

Flattens the entire matrix into a single row-major continuous memory buffer.

```cpp
std::vector<double> A = {
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
};

// Access element at (i, j) in row-major order: index = i * cols + j
double val = A[i * K + j];
```

* **Pros:** Single heap allocation, predictable sequential memory layout, direct interoperability with BLAS/C interfaces, ideal base for cache tiling and SIMD vectorization.
* **Cons:** Manual index computation, requires passing row/column dimensions separately.

---

## Technical Comparison

| Feature | `vector<vector<double>>` | Contiguous `vector<double>` |
| :--- | :--- | :--- |
| **Allocations** | $M + 1$ heap allocations | $1$ heap allocation |
| **Memory Locality** | Segmented / Non-contiguous rows | Fully contiguous (Row-Major) |
| **Element Access** | Double indirection (`A[i][j]`) | Single offset computation (`A[i * N + j]`) |
| **Dimension Tracking** | Self-contained via `.size()` | Stored / passed explicitly |
| **Optimization Suitability** | Low (Cache misses, indirection) | High (Vectorization & cache tiling ready) |

---

## Function Signatures

```cpp
// Segmented 2D storage
void gemm_vector(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    std::vector<std::vector<double>>& C
);

// Contiguous 1D storage
void gemm_contiguous(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    size_t M,
    size_t N,
    size_t K
);
```

---

## Correctness & Validation

`test_gemm.cpp` verifies that both implementations correctly calculate the matrix product of known $2 \times 2$ matrices using standard C++ runtime assertions (`assert`).

### Verification Workflow

Given test matrices:

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

The expected product matrix $C = A \times B$ is:

$$C = \begin{bmatrix} 1(5) + 2(7) & 1(6) + 2(8) \\ 3(5) + 4(7) & 3(6) + 4(8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

Both functions are checked against these expected values:

```cpp
// Verification for vector<vector<double>>
assert(C[0][0] == 19);
assert(C[0][1] == 22);
assert(C[1][0] == 43);
assert(C[1][1] == 50);

// Verification for contiguous vector<double>
assert(C_contig[0] == 19);
assert(C_contig[1] == 22);
assert(C_contig[2] == 43);
assert(C_contig[3] == 50);
```

### Build and Run

Compile with optimization disabled (`-O0`) to isolate logic correctness from compiler reordering or loop unrolling:

```bash
# Compile
g++ -std=c++17 -Wall -Wextra -O0 \
    gemm_vector.cpp \
    gemm_contiguous.cpp \
    test_gemm.cpp \
    -o test_gemm

# Execute tests
./test_gemm
```

### Compiler Flags

* `-std=c++17`: Standard C++17 conformance.
* `-Wall -Wextra`: Enable comprehensive diagnostic warnings.
* `-O0`: Disables optimizations to verify raw algorithmic logic during testing.
* `-o test_gemm`: Binary output target.
