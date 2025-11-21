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

## Detailed Method Explanations

This section expands on each implementation included in the benchmark. Each entry describes the approach, memory behavior, advantages, disadvantages, and when it is most appropriate to use.

- **Naive-ijkLoop**
	- Summary: Simple triple-nested loops (i, j, k) computing result[i][j] += A[i][k] * B[k][j].
	- Complexity: O(N^3). No extra memory.
	- Behavior: Good locality for A (row-major), poor locality for B (column-wise reads) — many cache misses.
	- Pros: Minimal code, baseline for correctness and timing.
	- Cons: Very poor cache utilization and no vectorization or parallelism.

- **BlockTiled-CacheAware**
	- Summary: Divide matrices into BLOCK_SIZE×BLOCK_SIZE tiles and compute per-tile to increase cache reuse.
	- Complexity: O(N^3) but much better constant factors due to reuse.
	- Behavior: Small working set per tile fits in cache, greatly reducing misses.
	- Pros: Large practical speedups, simple to combine with SIMD/threading.
	- Cons: Sensitive to BLOCK_SIZE; must handle tail blocks when N % BLOCK_SIZE != 0.

- **SIMD-AVX2-Transposed**
	- Summary: Create a transposed copy of B, then use AVX2 intrinsics to vectorize inner loops (processing multiple floats per instruction).
	- Complexity: O(N^3) compute; plus O(N^2) for the transpose.
	- Behavior: After transpose, both operands are read row-major in inner loops enabling wide, contiguous vector loads.
	- Pros: Strong speedups from SIMD; excellent memory throughput when aligned.
	- Cons: Extra memory and transpose cost; must handle tails and alignment.

- **RowColumn-Transposed**
	- Summary: Transpose B and compute row×row dot products (no SIMD required), improving locality.
	- Complexity: O(N^3) plus O(N^2) transpose.
	- Behavior: Better locality for inner-dot operations; easier to reason about than vectorized code.
	- Pros: Simple and effective locality improvement; portable.
	- Cons: Extra memory and transpose time; less raw throughput than SIMD versions.

- **Scalar-LoopUnrolled**
	- Summary: Scalar implementation with manual loop unrolling and small register blocking to reduce loop overhead and increase ILP.
	- Complexity: O(N^3).
	- Behavior: No extra big buffers; lower-level optimizations improve instruction throughput.
	- Pros: Portable and helpful when SIMD is unavailable.
	- Cons: Less performance than SIMD; more complex code.

- **Parallel-SIMD-AVX2**
	- Summary: Combine multithreading (partition by rows/blocks) with AVX2 vectorized inner loops; typically share a transposed B.
	- Complexity: O(N^3) total; wall-clock improves roughly by number of threads (subject to memory bandwidth).
	- Behavior: Each thread works on disjoint output ranges, improving locality per thread.
	- Pros: Best throughput on AVX2 multicore systems.
	- Cons: More complex (threading, partitioning); watch for false sharing and NUMA effects.

- **Parallel-Scalar-LoopUnrolled**
	- Summary: Threaded version of the scalar unrolled algorithm: threads process distinct ranges but inner compute remains scalar/unrolled.
	- Complexity: O(N^3) total; benefits from multiple cores.
	- Behavior: Lower memory overhead than SIMD+transpose variants.
	- Pros: Portable parallel speedup where SIMD is not available or undesirable.
	- Cons: Lower peak throughput than SIMD variants.

- **SIMD-AVX2-Direct**
	- Summary: Vectorize with AVX2 but operate on B directly (no global transpose); may use gather-like patterns or careful block loads.
	- Complexity: O(N^3).
	- Behavior: Avoids O(N^2) transpose memory cost but often suffers from non-contiguous loads.
	- Pros: Saves memory and transpose time.
	- Cons: Potentially lower SIMD efficiency due to unaligned/non-contiguous loads; more complex handling for tails.

- **Parallel-SIMD-Direct**
	- Summary: Multithreaded variant of SIMD-AVX2-Direct: threads run vectorized computation without creating a global transpose.
	- Complexity: O(N^3) total.
	- Behavior: Eliminates transpose but increases concurrent memory streams; memory bandwidth may become a bottleneck.
	- Pros: Useful when avoiding extra memory is important.
	- Cons: Scalability depends on memory subsystem; can be bandwidth-limited.

- **BlockLocal-StackTranspose**
	- Summary: For each tile pair, transpose a small sub-block of B into a stack-allocated aligned buffer, then compute using the local transposed block.
	- Complexity: O(N^3) with only small per-tile transposition overhead.
	- Behavior: Avoids allocating a full B^T while achieving most locality benefits of a full transpose.
	- Pros: Good balance of memory use and locality; avoids large O(N^2) temporary.
	- Cons: Requires careful block sizing; stack usage and edge handling need attention.

If you'd like, I can move these items into a dedicated section with short pseudocode or diagrams for the trickier methods (block tiling, local stack transpose), or add notes about numeric reproducibility across methods and how to tune `BLOCK_SIZE` and thread counts for your machine.

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
