# CPU Optimized GEMM (Loop Reordering)

An optimized CPU implementation of General Matrix-Matrix Multiplication (GEMM) in C++ focusing on hardware cache locality through loop permutation ($i \to k \to j$).

---

## Overview

The mathematical operation computes:

$$C = A \times B$$

Where:
* $A$ is an $M \times K$ matrix
* $B$ is a $K \times N$ matrix
* $C$ is an $M \times N$ matrix

Each element $C_{ij}$ is computed as:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$$

The algorithm retains its $\mathcal{O}(M \times N \times K)$ computational complexity while fundamentally improving data access patterns for modern CPU cache hierarchies.

---

## Loop Permutation Optimization

Both implementations operate on contiguous row-major `std::vector<double>` buffers, but they traverse memory differently:

### Naive Implementation ($i \to j \to k$)

```cpp
for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < K; ++k) {
            C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}
```

### Optimized Implementation ($i \to k \to j$)

```cpp
for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
        double r = A[i * K + k]; // Reused across inner loop
        for (size_t j = 0; j < N; ++j) {
            C[i * N + j] += r * B[k * N + j];
        }
    }
}
```

---

## Why Reorder the Loops?

In C++, matrices flattened into a 1D buffer follow **row-major order**, meaning consecutive elements of a row reside next to each other in physical memory.

* **Sequential Access in $B$ and $C$:** Making $j$ the innermost loop index guarantees that elements `B[k * N + j]` and `C[i * N + j]` are accessed with unit stride ($1$), maximizing CPU cache line utilization and hardware prefetching.
* **Temporal Locality in $A$:** The scalar element `A[i * K + k]` remains constant throughout the entire inner $j$ loop, allowing the CPU to keep it pinned in a register rather than fetching it from memory repeatedly.

```text
Row-major layout traversal in the inner j-loop:

B[k][0] ──► B[k][1] ──► B[k][2] ──► B[k][3] ...  (Contiguous unit stride)
C[i][0] ──► C[i][1] ──► C[i][2] ──► C[i][3] ...  (Contiguous unit stride)
```

---

## Traversal Comparison

| Metric | Naive ($i \to j \to k$) | Optimized ($i \to k \to j$) |
| :--- | :--- | :--- |
| **Matrix $A$ Access** | Sequential (`A[i * K + k]`) | **Register Reused** (`A[i * K + k]` invariant in $j$) |
| **Matrix $B$ Access** | Strided / Non-contiguous (`B[k * N + j]`) | **Sequential / Unit Stride** (`B[k * N + j]`) |
| **Matrix $C$ Access** | Repeated write to same scalar (`C[i * N + j]`) | **Sequential / Unit Stride** (`C[i * N + j]`) |
| **Cache Line Utilization** | Poor (stride across row boundaries) | **Optimal (full cache lines consumed)** |

---

## Data Representation & Function Signature

```cpp
void gemm_optimized(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    size_t M,
    size_t N,
    size_t K
);
```

* Matrices are stored as flat 1D vectors: `A(M * K)`, `B(K * N)`, and `C(M * N)`.
* Dimensions $M$, $N$, and $K$ are passed explicitly to support row-major offset calculations: `(row * cols + col)`.

---

## Correctness & Validation

`test_gemm.cpp` verifies the output against a known $2 \times 2$ matrix multiplication:

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \implies C = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

```cpp
assert(C[0] == 19);
assert(C[1] == 22);
assert(C[2] == 43);
assert(C[3] == 50);
```

### Build and Run

```bash
# Verify baseline logic without compiler optimizations
g++ -std=c++17 -Wall -Wextra -O0 gemm_optimized.cpp test_gemm.cpp -o test_gemm
./test_gemm

# Verify correctness with compiler optimization enabled
g++ -std=c++17 -Wall -Wextra -O3 gemm_optimized.cpp test_gemm.cpp -o test_gemm
./test_gemm
```

**Expected output:**
```text
All tests passed!
```

---

## Project Roadmap

This stage focuses strictly on **loop permutation and cache locality**. Subsequent CPU optimization stages include:

1. **Blocked / Tiled GEMM:** Explicit matrix tiling to fit L1/L2/L3 caches.
2. **SIMD Vectorization:** Explicit vector instructions (AVX2 / AVX-512 / FMA).
3. **Multithreading:** Work-sharing parallel loops with OpenMP.
