# Optimized Matrix Multiplication 

This repository contains a C implementation of matrix multiplication with various optimization techniques, including naive, SIMD (AVX2), cache-blocked, multi-threaded, and combined approaches. The code is designed to demonstrate performance improvements for matrix multiplication on modern CPUs.

## Features

- **Naive Implementation**: Basic matrix multiplication for reference.
- **AVX2 SIMD**: Uses Intel AVX2 instructions for vectorized operations.
- **Cache Blocking**: Optimizes memory access patterns to improve cache utilization.
- **Multi-Threading**: Parallelizes computation across multiple threads using POSIX threads.
- **Combined Optimizations**: Combines SIMD, cache blocking, and multi-threading for maximum performance.
- **NDArray Structure**: Flexible data structure for n-dimensional arrays with shape and stride support.

## Requirements

- GCC or compatible compiler
- CPU with AVX2 and FMA support
- POSIX-compliant system (for pthreads)

## Building

1. Clone the repository:
   ```bash
   git clone https://github.com/flyme2mars/optimized-matmul.git
   cd optimized-matmul
   ```

2. Build the project using the provided Makefile:
   ```bash
   make
   ```

3. Run the executable:
   ```bash
   ./matmul
   ```

## Usage

The `main` function in `matmul.c` initializes two matrices `A` (2000x3000) and `B` (3000x1200), performs matrix multiplication using each implementation, and prints the execution time for each method. The output matrix `C` is of size 2000x1200.

To modify matrix sizes or other parameters, edit the `shape_A`, `shape_B`, and `shape_C` arrays in the `main` function. Adjust block sizes (`bm`, `bn`, `bk`) or the number of threads (`NUM_THREADS`) as needed.

Example output:
```
Naive time: 7.765228
SIMD time: 4.446292
Cache blocked time: 3.783179
Cache blocked + SIMD time: 0.806368
Multi-threaded time: 1.124838
Multi-threaded + SIMD + Blocked time: 0.113685
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## Notes

- The code assumes matrices are stored in row-major order.
- Block sizes (18x18x18) and thread count (8) are set for demonstration and may need tuning for optimal performance on different hardware.
- Ensure proper alignment for AVX2 operations using `_mm_malloc` and `_mm_free`.
