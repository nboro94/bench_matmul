# Matrix Multiplication Benchmark

This project implements and benchmarks various matrix multiplication algorithms with different optimization techniques to demonstrate the impact of cache optimization, SIMD instructions, and multithreading on performance.

## Overview

Matrix multiplication is a fundamental operation in many computational fields including computer graphics, machine learning, and scientific computing. This benchmark compares different implementation strategies:

- Standard naive multiplication
- Cache-optimized multiplication using blocking/tiling
- Matrix transposition for better cache locality
- SIMD optimization using AVX2 intrinsics
- Multithreaded parallelization
- Combinations of the above techniques

## Features

- Automatic optimal block size determination based on CPU cache
- Automatic thread count optimization based on available CPU cores
- Multiple benchmark iterations for stable measurements
- Comprehensive performance comparison between implementations
- Correctness verification across all implementations
- Memory leak detection and reporting
- Detailed logging for debugging and analysis

## Requirements

- C++11 compatible compiler
- CPU with AVX2 instruction support
- CMake 2.8 or higher
- Linux environment (for CPU cache detection)

## Building

```bash
mkdir build
cd build
cmake ..
make -j
```

## Usage

Run the executable with an optional parameter to specify the matrix dimension:

```bash
./bench-matmul [dimension]
```

Where:
- `dimension`: Optional integer specifying the size of the square matrices (default: 512)

Example:
```bash
./bench-matmul 1024  # Run with 1024x1024 matrices
```

### Automated Benchmarking

You can run benchmarks for multiple matrix sizes automatically using the provided Python script:

```bash
python3 run_benchmarks.py
```

This script runs the benchmark with matrix sizes 64, 128, 512, 1024, 2048, 4096, and 8192, saving the output of each run to a separate log file in the `logs` directory.

You can specify a custom path to the executable:

```bash
python3 run_benchmarks.py --executable /path/to/bench-matmul
```

## Optimization Techniques

### Cache Blocking/Tiling
Improves cache utilization by processing the matrix in smaller blocks that fit in the CPU cache.

### Matrix Transposition
Transposes one of the matrices to ensure both matrices are accessed in row-major order, improving cache locality.

### SIMD (AVX2)
Uses Intel AVX2 instructions to process multiple elements simultaneously, increasing throughput.

### Multithreading
Distributes the workload across multiple CPU cores for parallel execution.

### Loop Reorganization
Reorganizes loop nesting order and adds manual loop unrolling to improve instruction-level parallelism.

## Output Interpretation

The program outputs a performance comparison table that includes:

- Execution time in milliseconds for each implementation
- Relative speedup compared to the baseline implementation
- Identification of the fastest implementation

Additionally, the program verifies that all implementations produce the same results to ensure correctness.

## Memory Management

The program uses a contiguous memory allocation strategy for matrices to improve cache performance. Memory allocation and deallocation are tracked to detect any memory leaks.

## License

MIT License

Copyright (c) 2025 Nishanta Boro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
