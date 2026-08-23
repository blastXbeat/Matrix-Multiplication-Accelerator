# CPU Blocked GEMM (Tiled Matrix Multiplication)

An optimized CPU implementation of General Matrix-Matrix Multiplication (GEMM) in C++ using cache blocking (tiling) and loop reordering ($ii \to kk \to jj \to i \to k \to j$) to maximize L1/L2/L3 cache data reuse.

---

## Overview

The GEMM operation computes:

$$C = A \times B$$

Where:
* $A$ is an $M \times K$ matrix
* $B$ is a $K \times N$ matrix
* $C$ is an $M \times N$ matrix

Each element $C_{ij}$ is defined as:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$$

The computational complexity remains $\mathcal{O}(M \times N \times K)$. Cache blocking does not reduce arithmetic operations; instead, it reduces cache misses by keeping active working tiles within CPU cache levels (L1/L2/L3).

---

## Why Blocked GEMM?

A standard matrix multiplication repeatedly traverses large memory buffers that exceed the CPU cache capacity, leading to cache eviction and redundant memory bus traffic.

Blocked GEMM partitions matrices into sub-matrices of size `BLOCK_SIZE × BLOCK_SIZE`:

```text
       Matrix A                   Matrix B
┌──────────┬──────────┐       ┌──────────┬──────────┐
│ A(0, 0)  │ A(0, 1)  │       │ B(0, 0)  │ B(0, 1)  │
├──────────┼──────────┤   ×   ├──────────┼──────────┤
│ A(1, 0)  │ A(1, 1)  │       │ B(1, 0)  │ B(1, 1)  │
└──────────┴──────────┘       └──────────┴──────────┘
```

The product is computed by accumulating block multiplications:

$$C_{\text{block}} += A_{\text{block}} \times B_{\text{block}}$$

This ensures that the working set fits into high-speed CPU caches, maximizing spatial and temporal reuse.

---

## Algorithm & Loop Structure

The algorithm uses 6 nested loops: 3 outer loops to iterate over tile coordinates, and 3 inner loops to compute the matrix product within the tile.

```text
ii (row tile of A and C)
 └── kk (shared dimension tile)
      └── jj (column tile of B and C)
           └── i (row index within tile)
                └── k (shared index within tile)
                     └── j (column index within tile)
```

### Core Implementation

```cpp
void gemm_blocked(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    size_t M,
    size_t N,
    size_t K,
    size_t BLOCK_SIZE
) {
    for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {

                // Handle non-divisible matrix boundary conditions
                size_t i_max = std::min(ii + BLOCK_SIZE, M);
                size_t k_max = std::min(kk + BLOCK_SIZE, K);
                size_t j_max = std::min(jj + BLOCK_SIZE, N);

                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        double a = A[i * K + k]; // Reused across inner j loop
                        for (size_t j = jj; j < j_max; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```

### Why the Inner $i \to k \to j$ Ordering?

* **Row-Major Unit Stride:** With $j$ as the innermost loop, accesses to `B[k * N + j]` and `C[i * N + j]` proceed sequentially through memory, taking full advantage of cache lines and CPU hardware prefetchers.
* **Register Pinning:** The scalar `a = A[i * K + k]` is loaded once and kept in a CPU register across the entire inner $j$ loop.

---

## Handling Arbitrary Matrix Dimensions

Matrix dimensions do not need to be multiples of `BLOCK_SIZE`. The loop bounds use `std::min` to handle edge tiles gracefully without requiring matrix padding:

```cpp
size_t i_max = std::min(ii + BLOCK_SIZE, M);
size_t k_max = std::min(kk + BLOCK_SIZE, K);
size_t j_max = std::min(jj + BLOCK_SIZE, N);
```

For instance, if $M = 100$ and $\text{BLOCK\_SIZE} = 32$, blocks will span ranges $[0, 32)$, $[32, 64)$, $[64, 96)$, and the final remnant block $[96, 100)$.

---

## Interface Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `A` | `const std::vector<double>&` | Input matrix buffer of size $M \times K$ |
| `B` | `const std::vector<double>&` | Input matrix buffer of size $K \times N$ |
| `C` | `std::vector<double>&` | Output matrix buffer of size $M \times N$ |
| `M` | `size_t` | Number of rows in $A$ and $C$ |
| `N` | `size_t` | Number of columns in $B$ and $C$ |
| `K` | `size_t` | Number of columns in $A$ / rows in $B$ |
| `BLOCK_SIZE` | `size_t` | Tile dimension along $M$, $N$, and $K$ |

---

## Correctness Verification

The test harness (`test_gemm.cpp`) verifies computation against known analytical values with a floating-point tolerance $(\epsilon = 10^{-9})$:

$$\text{abs}(C[i \times N + j] - C_{\text{expected}}[i \times N + j]) < 10^{-9}$$

For a $4 \times 4$ verification case with $\text{BLOCK\_SIZE} = 2$:

$$A = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 3 & 4 & 5 \\ 3 & 4 & 5 & 6 \\ 4 & 5 & 6 & 7 \end{bmatrix}, \quad B = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 3 & 4 & 5 \\ 3 & 4 & 5 & 6 \\ 4 & 5 & 6 & 7 \end{bmatrix} \implies C = \begin{bmatrix} 30 & 40 & 50 & 60 \\ 40 & 54 & 68 & 82 \\ 50 & 68 & 86 & 104 \\ 60 & 82 & 104 & 126 \end{bmatrix}$$

---

## Build and Run

### Standard Optimized Build

```bash
g++ -std=c++17 -Wall -Wextra -O3 gemm.cpp test_gemm.cpp -o test_gemm
./test_gemm
```

### Single-File Build (if combined)

```bash
g++ -std=c++17 -Wall -Wextra -O3 gemm.cpp -o test_gemm
./test_gemm
```

### Debug Build

```bash
g++ -std=c++17 -Wall -Wextra -g gemm.cpp test_gemm.cpp -o test_gemm
./test_gemm
```

**Expected Output:**
```text
Test passed!
```

---

## Architectural Comparison

| Stage | Loop Structure | Key Mechanism | Primary Bottleneck Addressed |
| :--- | :--- | :--- | :--- |
| **1. Naive** | $i \to j \to k$ | Basic 3-loop indexing | Correctness baseline |
| **2. Loop-Optimized** | $i \to k \to j$ | Unit stride on $B$ and $C$, register reuse of $A$ | Memory access stride & cache line waste |
| **3. Blocked (Current)** | $ii \to kk \to jj \to i \to k \to j$ | Sub-matrix tiling with boundary clamping | Cache capacity misses (L1/L2 data reuse) |

---

## Block Size Tuning Guidelines

Optimal `BLOCK_SIZE` depends on CPU L1/L2/L3 cache sizes. Recommended sweeps for benchmark experiments on $1024 \times 1024$ matrices:

$$\text{BLOCK\_SIZE} \in \{8, 16, 32, 64, 128, 256\}$$

A tile should satisfy the constraint:

$$3 \times (\text{BLOCK\_SIZE})^2 \times 8 \text{ bytes} \le \text{Cache Capacity (L1/L2)}$$
