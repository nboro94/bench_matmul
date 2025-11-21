 # bench_matmul — Matrix Multiplication Benchmark

 Concise, readable documentation for building, running, and comparing
 the matrix multiplication implementations included in this repository.

 ## Quickstart

 1. Build the project:

    ```bash
    mkdir -p build && cd build
    cmake ..
    make -j
    ```

 2. Run the binary for a single size (default N=512):

    ```bash
    ./bench-matmul [N]
    ```

 3. Or run the automated runner to execute many sizes and capture logs:

    ```bash
    python3 run_benchmarks.py --executable ./build/bench-matmul
    ```

 ## What this repository contains

- A C++ benchmarking program (`bench-matmul`) implementing multiple matrix
  multiplication algorithms (naive, tiled, SIMD, threaded, etc.).
- A Python helper (`run_benchmarks.py`) to run batches, capture logs, and
  compare implementations across multiple sizes.

 ## Building requirements

- C++11 compiler (GCC/Clang)
- CMake
- Linux (the program performs simple cache-detection via `/sys`)
- Optional: CPU with AVX2 to exercise vectorized code paths

 ## Binary usage (quick reference)

```text
./bench-matmul [N] [--list] [--run=name1,name2] [--baseline=name]
```

- `N` — matrix dimension (optional, default 512)
- `--list` — print available implementations and exit (fast, no allocation)
- `--run` — run only the named implementations (comma-separated)
- `--baseline` — request a specific baseline name (used for reporting)

Examples:

```bash
# list methods
./bench-matmul --list

# run two specific implementations for N=1024
./bench-matmul 1024 --run=BlockTiled-CacheAware,SIMD-AVX2-Transposed

# force baseline selection
./bench-matmul 512 --run=BlockTiled-CacheAware --baseline=Naive-ijkLoop
```

## Automated runner (`run_benchmarks.py`)

Purpose: run multiple matrix sizes, capture stdout to timestamped logs, run
comparisons, and parallelize runs across sizes.

Important CLI options
- `--executable PATH` — path to `bench-matmul` (default `./bench-matmul`).
- `--run NAME1,NAME2` — methods to pass through to the binary (aliases supported).
- `--baseline NAME` — an optional baseline alias/name forwarded to the binary.
- `--sizes S1,S2` — comma-separated sizes to benchmark (defaults: 64,128,512,1024).
- `--compare M1,M2,...` — run and compare the listed methods (at least two).
- `-j N` / `--jobs N` — run up to `N` sizes concurrently (default `1`).

Examples

```bash
# run default sizes with packaged binary
python3 run_benchmarks.py

# run only two implementations for small sizes, two concurrent jobs
python3 run_benchmarks.py --executable ./build/bench-matmul --run=naive,tiled --sizes=64,128 -j 2

# compare three implementations (aliases allowed)
python3 run_benchmarks.py --executable ./build/bench-matmul --compare=tiled-par,tiled,naive --sizes=64 -j 1

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

How `--compare` works
- The runner normalizes aliases to full method names and runs each method
  for each requested size. It parses the benchmark's output to extract the
  reported average execution time and prints a compact table of times and
  relative speedups (by default relative to the first method listed in the
  `--compare` argument). `--baseline` can be forwarded to the binary if you
  want the binary to use a particular baseline for its internal reporting.

## Alias mapping (supported by `run_benchmarks.py`)

You can use these shortcuts when calling the runner; the script maps them to
the full implementation names:

| Alias | Full name |
|-------|-----------|
| `naive` | `Naive-ijkLoop` |
| `tiled` | `BlockTiled-CacheAware` |
| `naive-par` | `Naive-ijkLoop-Parallel` |
| `tiled-par` | `BlockTiled-CacheAware-Parallel` |
| `avx2` | `SIMD-AVX2-Transposed` |
| `avx2direct` | `SIMD-AVX2-Direct` |
| `transposed` | `RowColumn-Transposed` |
| `scalar` | `Scalar-LoopUnrolled` |
| `par-avx2` | `Parallel-SIMD-AVX2` |
| `par-scalar` | `Parallel-Scalar-LoopUnrolled` |
| `par-avx2-direct` | `Parallel-SIMD-Direct` |
| `local` | `BlockLocal-StackTranspose` |

## Implementations (short descriptions)

 `Naive-ijkLoop-Parallel` — naive triple-loop implementation parallelized across rows.
- `Scalar-LoopUnrolled` — scalar unrolled inner loops for ILP.
- `Parallel-SIMD-AVX2` — threaded + AVX2 vectorization.
- `Parallel-Scalar-LoopUnrolled` — threaded scalar unrolled implementation.
- `SIMD-AVX2-Direct` — AVX2 without pre-transposition (direct loads).
- `Parallel-SIMD-Direct` — threaded variant of AVX2 direct loads.
- `BlockLocal-StackTranspose` — tile-local transpose into stack buffer.

## Output & verification

- The binary verifies correctness by comparing each selected implementation
  against the chosen baseline.
- The benchmark emits a performance table listing average execution times
  and relative speedups; the runner aggregates and prints concise comparison
  tables when `--compare` is used.

## Memory and diagnostics

- Matrices use contiguous allocations (single data block + row pointers)
  for better cache behaviour.
- The program logs allocations/deallocations and reports a warning if
  allocations do not match deallocations at exit.

## License

MIT License — Copyright (c) 2025 Nishanta Boro

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
