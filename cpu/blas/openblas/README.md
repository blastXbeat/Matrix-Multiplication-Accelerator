# OpenBLAS GEMM

## Overview

This implementation performs matrix multiplication using **OpenBLAS**, a highly optimized implementation of the BLAS standard.

The matrix multiplication operation is:

```text
C = A × B
```

More generally, BLAS GEMM computes:

```text
C = αAB + βC
```

For this implementation:

```text
α = 1.0
β = 0.0
```

Therefore:

```text
C = 1.0 × AB + 0.0 × C

C = AB
```

This implementation is part of the CPU optimization study and provides a comparison point against manually implemented approaches such as naive GEMM, loop optimization, cache blocking, OpenMP, and SIMD.

---

# What is BLAS?

**BLAS** stands for:

> Basic Linear Algebra Subprograms

BLAS is a standard library interface for common linear algebra operations.

These operations are generally divided into three levels:

### Level 1

Vector-vector operations.

Examples:

* Vector addition
* Dot product

### Level 2

Matrix-vector operations.

Example:

```text
y = Ax
```

### Level 3

Matrix-matrix operations.

Example:

```text
C = AB
```

GEMM belongs to **BLAS Level 3**.

---

# What is GEMM?

GEMM stands for:

> General Matrix Multiply

The general GEMM operation is:

```text
C = αAB + βC
```

For matrices:

```text
A → M × K

B → K × N

C → M × N
```

The multiplication is valid because the inner dimensions match:

```text
(M × K) × (K × N) = (M × N)
```

Example:

```text
A = 2 × 3

B = 3 × 4

C = 2 × 4
```

---

# What is OpenBLAS?

OpenBLAS is an open-source, optimized implementation of BLAS.

It provides highly optimized routines for linear algebra operations, including matrix multiplication.

Internally, optimized BLAS implementations can use techniques such as:

* Cache blocking
* Matrix packing
* SIMD instructions
* CPU-specific microkernels
* Assembly-level optimizations
* Multithreading

This makes OpenBLAS useful as a high-performance reference implementation when comparing manually written GEMM implementations.

---

# `cblas_dgemm`

This implementation uses:

```cpp
cblas_dgemm()
```

The name can be understood as:

```text
cblas  → C interface to BLAS
d      → Double precision
gemm   → General Matrix Multiply
```

Therefore:

```cpp
cblas_dgemm()
```

performs double-precision matrix multiplication.

---

# Matrix Storage

This implementation uses:

```cpp
CblasRowMajor
```

Matrices are stored in row-major order.

For example:

```text
A =

1 2 3
4 5 6
```

The matrix is stored in memory as:

```text
1 2 3 4 5 6
```

The implementation uses:

```cpp
std::vector<double>
```

for matrix storage.

---

# GEMM Parameters

The general function structure is:

```cpp
cblas_dgemm(
    layout,
    transposeA,
    transposeB,
    M,
    N,
    K,
    alpha,
    A,
    lda,
    B,
    ldb,
    beta,
    C,
    ldc
);
```

The implementation uses:

```cpp
cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    M,
    N,
    K,
    1.0,
    A.data(),
    K,
    B.data(),
    N,
    0.0,
    C.data(),
    N
);
```

---

# Parameter Explanation

## Matrix Layout

```cpp
CblasRowMajor
```

Specifies that matrices are stored row by row.

---

## Transpose Options

```cpp
CblasNoTrans
```

means the matrix is used without transposition.

This implementation uses:

```cpp
CblasNoTrans,
CblasNoTrans
```

Therefore:

```text
C = A × B
```

Neither matrix is transposed.

---

## Matrix Dimensions

The matrices have the following dimensions:

```text
A → M × K

B → K × N

C → M × N
```

Therefore, GEMM receives:

```cpp
M
N
K
```

---

## Alpha

```cpp
1.0
```

This corresponds to:

```text
α = 1.0
```

---

## Beta

```cpp
0.0
```

This corresponds to:

```text
β = 0.0
```

Therefore:

```text
C = 1.0 × AB + 0.0 × C
```

which becomes:

```text
C = AB
```

---

# Matrix Data

The matrix vectors are passed to OpenBLAS using:

```cpp
A.data()
B.data()
C.data()
```

For example:

```cpp
A.data()
```

returns a pointer to the underlying data stored in the vector.

OpenBLAS uses these pointers directly to access the matrix data.

The output matrix `C` is modified directly.

After `cblas_dgemm()` returns:

```text
C already contains the multiplication result.
```

No additional multiplication loops are required.

---

# Leading Dimensions

The parameters:

```text
lda
ldb
ldc
```

are called leading dimensions.

For row-major matrices:

```text
A → M × K → K elements per row

B → K × N → N elements per row

C → M × N → N elements per row
```

Therefore:

```cpp
lda = K;
ldb = N;
ldc = N;
```

This is why the implementation uses:

```cpp
A.data(), K
B.data(), N
C.data(), N
```

---

# Input Validation

Before calling OpenBLAS, the matrix sizes are checked.

Expected sizes:

```text
A.size() = M × K

B.size() = K × N

C.size() = M × N
```

If the dimensions do not match, the function throws:

```cpp
invalid_argument
```

Example:

```cpp
if (A.size() != M * K ||
    B.size() != K * N ||
    C.size() != M * N) {

    throw invalid_argument("Matrix dimensions do not match.");
}
```

This prevents invalid matrix data from being passed to the GEMM routine.

---

# Installation

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install libopenblas-dev
```

This installs the OpenBLAS development library and required headers.

The GEMM implementation requires:

```cpp
#include <cblas.h>
```

---

# Compilation

For correctness testing:

```bash
g++ -std=c++17 -Wall -Wextra -O0 gemm.cpp test_gemm.cpp -lopenblas -o test_gemm
```

### Compilation Flags

```text
-std=c++17
```

Uses the C++17 standard.

```text
-Wall
```

Enables common compiler warnings.

```text
-Wextra
```

Enables additional compiler warnings.

```text
-O0
```

Disables compiler optimizations.

This is useful during correctness testing because the focus is on verifying functionality.

```text
-lopenblas
```

Links the OpenBLAS library.

```text
-o test_gemm
```

Names the output executable `test_gemm`.

---

# Running

After compilation:

```bash
./test_gemm
```

The test program should verify that the OpenBLAS result matches the expected matrix multiplication result.

---

# Optimized Compilation

After correctness has been verified, compile with optimization enabled:

```bash
g++ -std=c++17 -Wall -Wextra -O3 gemm.cpp test_gemm.cpp -lopenblas -o test_gemm
```

The flag:

```text
-O3
```

enables aggressive compiler optimizations.

`-O3` is appropriate when performing performance measurements or benchmarking.

---

# Example

Given:

```text
A =

1 2
3 4
```

and:

```text
B =

5 6
7 8
```

The result is:

```text
C = A × B
```

```text
C =

19 22
43 50
```

The calculation is:

```text
C[0][0] = (1 × 5) + (2 × 7) = 19

C[0][1] = (1 × 6) + (2 × 8) = 22

C[1][0] = (3 × 5) + (4 × 7) = 43

C[1][1] = (3 × 6) + (4 × 8) = 50
```

---

# Comparison With Previous Implementations

| Implementation | Main Optimization                        |
| -------------- | ---------------------------------------- |
| Naive GEMM     | Basic triple nested loops                |
| Optimized GEMM | Improved loop ordering and memory access |
| Blocked GEMM   | Improved cache utilization               |
| OpenMP         | CPU multithreading                       |
| AVX2           | 256-bit SIMD                             |
| AVX512         | 512-bit SIMD                             |
| OpenBLAS       | Highly optimized BLAS implementation     |

OpenBLAS acts as a high-performance reference point for the manually implemented CPU GEMM versions.

---

# Key Takeaways

* BLAS is a standard interface for linear algebra operations.
* GEMM is a BLAS Level 3 matrix-matrix multiplication operation.
* OpenBLAS is an optimized implementation of BLAS.
* `cblas_dgemm()` performs double-precision matrix multiplication.
* The general operation is:

```text
C = αAB + βC
```

* This implementation uses:

```text
α = 1.0
β = 0.0
```

resulting in:

```text
C = AB
```

* Matrices are stored using row-major layout.
* `A.data()`, `B.data()`, and `C.data()` provide OpenBLAS access to the matrix data.
* No manual multiplication loops are required.
* `-lopenblas` is required during compilation to link the OpenBLAS library.
* `-O0` is useful for correctness testing.
* `-O3` should be used for performance benchmarking.
