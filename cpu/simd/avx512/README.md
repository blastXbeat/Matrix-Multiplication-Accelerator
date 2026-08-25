
# AVX-512 SIMD GEMM

This directory contains AVX-512 implementations of matrix multiplication (GEMM).

## Difference from AVX2

AVX-512 uses **512-bit SIMD registers**, while AVX2 uses **256-bit registers**.

For `double` values:

- AVX2 processes **4 doubles** at once.
- AVX-512 processes **8 doubles** at once.

| Feature | AVX2 | AVX-512 |
|---|---:|---:|
| Register width | 256-bit | 512-bit |
| Doubles per vector | 4 | 8 |
| Vector type | `__m256d` | `__m512d` |

The implementation approach is similar to AVX2, but AVX-512 processes twice as many `double` values per SIMD instruction.

## Compilation

### Correctness testing

```bash
g++ -std=c++17 -Wall -Wextra -O0 -mavx512f -mfma gemm.cpp test_gemm.cpp -o test_gemm
````

### Performance testing

```bash
g++ -std=c++17 -Wall -Wextra -O3 -mavx512f -mfma gemm.cpp test_gemm.cpp -o test_gemm
```

## Run

```bash
./test_gemm
```

## Note

AVX-512 requires CPU support. Check using:

```bash
lscpu | grep -i avx
```

Look for `avx512f` in the CPU flags.


