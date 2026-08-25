# SIMD Matrix Multiplication

This directory contains SIMD-optimized implementations of matrix multiplication using **AVX2** and **AVX-512** instructions. Each implementation computes:

```text
C = A × B
```

where `A` is `M × K`, `B` is `K × N`, and `C` is `M × N`.

## What is SIMD?

**SIMD** stands for *Single Instruction, Multiple Data*. It lets one CPU instruction perform the same operation on several values at once. In matrix multiplication, SIMD is used to multiply and accumulate multiple adjacent elements of an output row in parallel.

For example, instead of four separate multiplications, a SIMD register can operate on four or more values in one instruction.

## Implementations

| Directory | Instruction set | Double-precision values per vector |
| --- | --- | --- |
| `avx2/` | AVX2 (256-bit) | 4 |
| `avx512/` | AVX-512 (512-bit) | 8 |

For single-precision `float` values, AVX2 processes 8 values per vector and AVX-512 processes 16.

Both implementations typically use fused multiply-add (FMA) instructions, which calculate:

```text
accumulator = a × b + accumulator
```

This is the core operation used to accumulate the dot products that form matrix multiplication.

## Notes

- Matrices are normally stored in row-major order.
- SIMD is most effective when processing consecutive elements in memory.
- Dimensions that are not a multiple of the vector width require a scalar tail path or masked vector operations.
- AVX-512 is not available on every processor; runtime hardware support is required.
- See the README in each subdirectory for implementation details, build commands, and correctness tests.

## Requirements

Compile each implementation with the appropriate target flags. Typical examples are:

```bash
# AVX2 implementation
g++ -O3 -mavx2 -mfma ...

# AVX-512 implementation
g++ -O3 -mavx512f -mfma ...
```

Use only instruction sets supported by the CPU running the program; otherwise it may terminate with an illegal-instruction error.
