# AVX2 SIMD GEMM

Two AVX2-based implementations of double-precision matrix multiplication (GEMM). They demonstrate how loop order, memory locality, and register accumulation affect CPU performance.

## What is AVX2?

**AVX2 (Advanced Vector Extensions 2)** is a CPU instruction set extension that enables **SIMD (Single Instruction, Multiple Data)** operations.

AVX2 uses **256-bit vector registers**, allowing multiple values to be processed with a single instruction. For `double` values:

```text
256 bits ÷ 64 bits = 4 doubles
```

## What it computes

```text
C = A × B

A: M × K
B: K × N
C: M × N
```

Matrices use row-major `std::vector<double>` storage:

```cpp
A[i * K + k]
B[k * N + j]
C[i * N + j]
```

## Implementations

| File | Loop order | Approach |
| --- | --- | --- |
| `gemm_avx2_ikj.cpp` | `i → k → j` | Streams through rows of `B` and `C`; updates `C` for every `k`. |
| `gemm_avx2_ijk.cpp` | `i → j → k` | Keeps a four-element `C` chunk in an AVX register during the complete `K` accumulation. |

Both vectorize along the `j` dimension, processing four `double` values at once.

## AVX2 operations

| Intrinsic | Use |
| --- | --- |
| `_mm256_set1_pd` | Broadcasts one `A[i][k]` value to four lanes. |
| `_mm256_loadu_pd` | Loads four consecutive, possibly unaligned, `double` values. |
| `_mm256_fmadd_pd` | Performs `a * b + c` as a fused multiply-add. |
| `_mm256_storeu_pd` | Stores four `double` values without an alignment requirement. |

`__m256d` is a 256-bit register, so it holds four 64-bit doubles.

## Build and run

Build a correctness-oriented version:

```bash
g++ -std=c++17 -Wall -Wextra -O0 -mavx2 -mfma gemm_avx2.cpp test_gemm.cpp -o test_gemm
./test_gemm
```

For performance testing, switch `-O0` to `-O3`:

```bash
g++ -std=c++17 -Wall -Wextra -O3 -mavx2 -mfma gemm_avx2.cpp test_gemm.cpp -o test_gemm
./test_gemm
```

> Adjust `gemm_avx2.cpp` to the implementation filename you want to compile, or adapt the command if your test harness builds both files.

## CPU requirement

The host CPU must support both AVX2 and FMA. Check on Linux with:

```bash
lscpu | grep -E 'avx2|fma'
```

Running this code on unsupported hardware can cause an illegal-instruction error.

## Tail handling

AVX2 handles columns in groups of four. When `N` is not divisible by four, the remaining columns are computed with scalar code:

```cpp
for (; j + 3 < N; j += 4) {
    // AVX2 path
}
for (; j < N; ++j) {
    // scalar tail
}
```

## Correctness

Validate each SIMD result against a scalar reference and compare with a tolerance, such as `1e-9`, rather than exact equality. Fused operations and different accumulation orders may produce tiny floating-point rounding differences.


