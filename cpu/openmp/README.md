# OpenMP GEMM

## Overview
This implementation accelerates General Matrix Multiplication (GEMM) using OpenMP shared-memory parallelism.
The implementation builds on the blocked GEMM approach by distributing independent row blocks across multiple CPU threads. This allows the computation to utilize multiple CPU cores while retaining cache-friendly blocking.

---

## Algorithm
For matrix multiplication:
$$C = A \times B$$

where:
* $A$ has dimensions $M \times K$
* $B$ has dimensions $K \times N$
* $C$ has dimensions $M \times N$

Each output element is computed as:
$$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \times B[k][j]$$

The computation is divided into blocks. OpenMP parallelizes the outer block loop:
```c
#pragma omp parallel for schedule(static)
```

Each thread processes different row blocks of the output matrix:
* **Thread 0** $\rightarrow$ Row blocks of $C$
* **Thread 1** $\rightarrow$ Different row blocks of $C$
* **Thread 2** $\rightarrow$ Different row blocks of $C$
* **Thread 3** $\rightarrow$ Different row blocks of $C$

Since each thread operates on different rows of $C$, the implementation avoids data races without requiring synchronization for individual matrix updates.

---

## Implementation Strategy
The implementation combines two optimization techniques:

1. **Cache blocking**
   * Matrices are processed in smaller blocks.
   * Improves cache locality.
   * Reduces unnecessary memory traffic.

2. **OpenMP parallelization**
   * Independent row blocks are distributed across CPU threads.
   * Allows multiple CPU cores to perform matrix multiplication simultaneously.

The loop structure is conceptually:
```text
for each row block ii
    parallelize across threads

    for each column block jj
        for each reduction block kk
            for each row i
                for each k
                    load A[i][k]

                    for each column j
                        C[i][j] += A[i][k] × B[k][j]
```

---

## Files
```text
cpu/openmp/
├── gemm_openmp.cpp
├── test_gemm.cpp
└── README.md
```

* `gemm_openmp.cpp` — OpenMP-based GEMM implementation
* `test_gemm.cpp` — Correctness test
* `README.md` — Documentation

---

## Compilation

### Correctness testing
Compile without compiler optimizations:
```bash
g++ -O0 -Wall -Wextra -fopenmp gemm_openmp.cpp test_gemm.cpp -o test_gemm
```

Run:
```bash
./test_gemm
```

### Performance build
For benchmarking, compile with optimization enabled:
```bash
g++ -O3 -Wall -Wextra -fopenmp gemm_openmp.cpp benchmark.cpp -o benchmark
```

*The `-O3` optimization level should be used for performance measurements, while `-O0` is useful during initial development and debugging.*

---

## Compiler Flags

| Flag | Purpose |
| :--- | :--- |
| `-O0` | Disable optimization for easier debugging |
| `-O3` | Enable aggressive compiler optimization for performance |
| `-Wall` | Enable common compiler warnings |
| `-Wextra` | Enable additional compiler warnings |
| `-fopenmp` | Enable OpenMP support |

---

## Thread Control
OpenMP automatically determines the number of threads unless configured otherwise.
The number of threads can be controlled using the `OMP_NUM_THREADS` environment variable.

Example:
```bash
OMP_NUM_THREADS=1 ./test_gemm
OMP_NUM_THREADS=2 ./test_gemm
OMP_NUM_THREADS=4 ./test_gemm
OMP_NUM_THREADS=8 ./test_gemm
```
This allows the implementation to be tested with different levels of CPU parallelism.

---

## Correctness Testing
The implementation should be verified by comparing the computed output matrix against the expected result.
Correctness tests should verify:
* Square matrices
* Rectangular matrices
* Different block sizes
* Small matrices
* Matrix dimensions that are not multiples of the block size

A numerical tolerance can be used when comparing floating-point values:
$$|\text{expected} - \text{actual}| < \text{tolerance}$$

---

## Expected Benefits
Compared with single-threaded GEMM implementations, OpenMP can provide:
* Multi-core CPU utilization
* Reduced execution time for sufficiently large matrices
* Better scalability across CPU cores

The actual speedup depends on:
* Matrix size
* Number of CPU cores
* Memory bandwidth
* Cache behavior
* OpenMP scheduling overhead

*Small matrices may not benefit significantly because thread-management overhead can outweigh the benefit of parallel execution.*

---

## Limitations
This implementation is still a custom GEMM kernel and does not include all optimizations used by production BLAS libraries.
Potential limitations include:
* Memory bandwidth saturation
* Thread synchronization and scheduling overhead
* Limited scaling at high thread counts
* Possible cache contention between threads
* No explicit SIMD intrinsics
