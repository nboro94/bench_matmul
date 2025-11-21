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

Command-line options
- `--list` : Print the available multiplication methods and exit (no heavy initialization).
- `--run=name1,name2,...` : Run only the named multiplication methods (comma-separated). If omitted, all methods are benchmarked. Names must match the implementation labels below; the `run_benchmarks.py` script supports short aliases.

Examples:
```bash
./bench-matmul --list
./bench-matmul 1024 --run=Naive-ijkLoop,BlockTiled-CacheAware
```

### Automated Benchmarking

You can run benchmarks for multiple matrix sizes automatically using the provided Python script:

```bash
python3 run_benchmarks.py
```

The script supports a few additional options to make batch runs easier:

- `--executable PATH` : path to the benchmark binary (default `./bench-matmul`).
- `--run NAME1,NAME2` : comma-separated list of methods to run (supports short aliases — see below). The script will forward these as `--run=` to the executable.
- `--sizes S1,S2` : comma-separated list of matrix sizes to run instead of the defaults.
- `--list` : query the executable for available methods and print them (script exits).
- `-j N / --jobs N` : run up to N sizes in parallel (concurrent jobs). Default is `1` (sequential).

Examples:

```bash
# Run the default suite with the bundled binary
python3 run_benchmarks.py

# Run only two methods for small sizes and two concurrent jobs
python3 run_benchmarks.py --executable ./build/bench-matmul --run=naive,tiled --sizes=64,128 -j 2

# Query the binary for available methods
python3 run_benchmarks.py --executable ./build/bench-matmul --list
```

The script saves each run's output to a timestamped log file under `logs` (or `logs_<n>` if `logs` already exists).

Alias mapping (supported by `run_benchmarks.py`)

You can use short, convenient aliases when calling `run_benchmarks.py`. These map to the full method labels used by the C++ binary:

 - `naive` -> `Naive-ijkLoop`
 - `tiled` -> `BlockTiled-CacheAware`
 - `avx2` -> `SIMD-AVX2-Transposed`
 - `avx2direct` -> `SIMD-AVX2-Direct`
 - `transposed` -> `RowColumn-Transposed`
 - `scalar` -> `Scalar-LoopUnrolled`
 - `par-avx2` -> `Parallel-SIMD-AVX2`
 - `par-scalar` -> `Parallel-Scalar-LoopUnrolled`
 - `par-avx2-direct` -> `Parallel-SIMD-Direct`
 - `local` -> `BlockLocal-StackTranspose`

If you prefer, you can also pass the full implementation names directly to `--run`.

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
