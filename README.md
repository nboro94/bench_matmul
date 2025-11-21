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

# forward baseline to the binary (useful when you want a specific baseline)
python3 run_benchmarks.py --baseline=naive --run=tiled-par,avx2 --sizes=64
```

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
