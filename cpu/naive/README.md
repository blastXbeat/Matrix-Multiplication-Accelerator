# CPU Naive GEMM

This directory contains the baseline CPU implementation of General Matrix-Matrix Multiplication (GEMM).

Two matrix storage approaches are implemented to study the effect of memory layout:

1. `vector<vector<double>>` — row-wise dynamic storage
2. `vector<double>` — single contiguous memory block

The mathematical GEMM operation is the same in both implementations:

[
C = A \times B
]

For:

* (A): (M \times K)
* (B): (K \times N)
* (C): (M \times N)

the computation is:

[
C_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}
]

Both implementations use the basic three-loop algorithm with (O(MNK)) computational complexity.

---

## Implementations

### 1. `vector<vector<double>>`

This implementation represents a matrix as a vector containing individual row vectors.

```cpp
vector<vector<double>> A = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};
```

The matrix can be accessed naturally using:

```cpp
A[i][j]
```

The dimensions can be obtained from the nested vectors:

```cpp
size_t rows = A.size();
size_t cols = A[0].size();
```

Conceptually:

```text
A
│
├── Row 0 → [1 2 3]
├── Row 1 → [4 5 6]
└── Row 2 → [7 8 9]
```

Each row is a separate `vector<double>` allocation.

### Advantages

* Simple and intuitive 2D indexing
* Easy to construct and understand
* Convenient for initial correctness development

### Limitations

* Rows are not guaranteed to be adjacent in memory
* Multiple dynamic allocations
* Less suitable for studying contiguous memory access and cache behavior
* Additional indirection when accessing elements

---

## 2. Contiguous `vector<double>`

The matrix is stored in one single 1D vector.

```cpp
vector<double> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
};
```

For a row-major matrix, element `(i, j)` is accessed using:

```cpp
A[i * cols + j]
```

For example, for a `3 × 3` matrix:

```text
Logical matrix:

[ 1  2  3 ]
[ 4  5  6 ]
[ 7  8  9 ]

Physical storage:

[ 1  2  3  4  5  6  7  8  9 ]
```

The matrix dimensions are therefore stored or passed separately:

```cpp
size_t M = 3;
size_t N = 3;
```

`A.size()` only returns the total number of elements:

```cpp
A.size() == M * N
```

It does not independently tell the program the number of rows and columns.

### Advantages

* All matrix elements occupy one contiguous memory region
* Better suited for predictable memory access
* Easier to use with low-level C/C++ interfaces
* Provides a better foundation for future cache, SIMD, and blocked implementations

### Limitations

* Requires explicit index calculation
* Dimensions must be tracked separately
* `A[i][j]` cannot be used directly with a normal `vector<double>`

---

## Comparison

| Property                       | `vector<vector<double>>`  | Contiguous `vector<double>` |
| ------------------------------ | ------------------------- | --------------------------- |
| Storage                        | Multiple row vectors      | One vector                  |
| Memory layout                  | Rows separately allocated | Fully contiguous            |
| Indexing                       | `A[i][j]`                 | `A[i * cols + j]`           |
| `A.size()`                     | Number of rows            | Total number of elements    |
| Column size                    | `A[0].size()`             | Stored separately           |
| Allocations                    | Multiple                  | One                         |
| Memory locality                | Less predictable          | Predictable                 |
| Ease of use                    | Higher                    | Lower                       |
| Hardware-oriented optimization | Less suitable             | Better foundation           |

---

# Correctness Testing

`test_gemm.cpp` is used to verify that both implementations produce the same correct result.

The test does not benchmark performance. It only checks correctness.

The testing flow is:

```text
              Same input matrices
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
     vector<vector>       contiguous vector
        GEMM                    GEMM
             │                   │
             ▼                   ▼
          Result A             Result B
             │                   │
             └─────────┬─────────┘
                       ▼
                    Compare
                       │
                  PASS / FAIL
```

A small known matrix multiplication should be tested first, followed by rectangular and random matrices.

For floating-point results, comparisons should eventually use a tolerance rather than direct equality.

Example:

```cpp
abs(result_a - result_b) < tolerance
```

---

# Building the Correctness Test

From the `cpu/naive` directory, compile all implementations and the test program together:

```bash
g++ -std=c++17 -Wall -Wextra -O0 \
    gemm_vector.cpp \
    gemm_contiguos.cpp \
    test_gemm.cpp \
    -o test_gemm
```

### Compiler options

* `-std=c++17` — compile using the C++17 standard
* `-Wall` — enable common compiler warnings
* `-Wextra` — enable additional warnings
* `-O0` — disable compiler optimization during correctness testing
* `-o test_gemm` — name the resulting executable `test_gemm`

Run the test:

```bash
./test_gemm
```

Expected output:

```text
All tests passed!
```

---

# Why `-O0`?

At this stage, the goal is **correctness**, not performance.

Optimization and benchmarking will be introduced later. Keeping optimization disabled makes this stage focused on verifying the implementation itself.

Do not use these correctness tests as the project's final performance benchmark.

---

# Function Interfaces

The two implementations use separate function names so they can be compiled and tested together.

Example:

```cpp
void gemm_vector(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C
);
```

and:

```cpp
void gemm_contiguos(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    size_t M,
    size_t N,
    size_t K
);
```

The `const` qualifier on `A` and `B` indicates that the GEMM function only reads the input matrices and does not modify them.

---
