# Benchmark Conclusions (All Methods, CPU + CUDA)

## Run Summary

- Executable: `build/Release/bench-matmul.exe`
- Methods: all methods listed by `--list`, including `CUDA-Naive`
- Requested sizes: `64, 128, 256, 512, 1024, 2048, 3072, 4096, 6000`
- Completed sizes: `64, 128, 256, 512, 1024, 2048, 3072, 4096`
- Incomplete size: `6000` (log file exists but is empty: `logs_1/matmul_bench_6000_all_20260213_021153_137953.log`)
- Iterations per method: 3

All completed logs passed:
- Correctness verification (`Verification successful`)
- Memory checks (`Memory check passed`)

## Fastest Method by Size

- `N=64`: `SIMD-AVX2-Direct` (0.024 ms)
- `N=128`: `BlockTiled-CacheAware-Parallel` (0.217 ms)
- `N=256`: `Parallel-SIMD-TBB` (0.573 ms)
- `N=512`: `Parallel-SIMD-TBB` (1.970 ms)
- `N=1024`: `Parallel-SIMD-TBB` (8.678 ms)
- `N=2048`: `CUDA-Naive` (45.633 ms)
- `N=3072`: `CUDA-Naive` (71.275 ms)
- `N=4096`: `CUDA-Naive` (129.009 ms)

## Key Performance Trend

There is a clear crossover:
- For small to medium matrices (`N <= 1024`), CPU optimized parallel methods win.
- For large matrices (`N >= 2048`), CUDA becomes fastest and scales better than CPU methods.

This is the central conclusion of this run.

## CUDA Behavior and Interpretation

CUDA timing pattern is consistent across large sizes:
- Iteration 1 is significantly slower than iterations 2/3.
- Causes: first-use GPU runtime setup, context creation, and warm-up overhead.
- Example (`N=4096`):
  - Iter 1: 190.912 ms
  - Iter 2: 97.375 ms
  - Iter 3: 98.741 ms
  - Reported average: 129.009 ms

Implication:
- For production throughput, steady-state CUDA performance is better than the reported average implies.
- For latency-sensitive one-shot calls, startup overhead must be included.

## CPU Family Findings

- `Parallel-SIMD-TBB` is the strongest CPU method overall for `N=256..1024`.
- `BlockTiled-CacheAware-Parallel` is consistently close to top CPU performance and wins at `N=128`.
- `SIMD-AVX2-Direct` is best only at tiny sizes (`N=64`), where thread/scheduling overhead dominates.
- Serial baselines become uncompetitive quickly as size increases.

## Scaling Notes

- Naive baseline growth is extreme (as expected for O(N^3)):
  - `N=1024`: 1552.042 ms
  - `N=2048`: 23080.346 ms
  - `N=3072`: 83141.683 ms
  - `N=4096`: 465839.861 ms
- CUDA relative speedup vs naive grows strongly with size:
  - `N=2048`: 505.78x
  - `N=3072`: 1166.50x
  - `N=4096`: 3610.90x

## Practical Recommendations

1. If matrix sizes are mostly below ~1500:
   - Prefer CPU path: `Parallel-SIMD-TBB`
   - Fallback: `BlockTiled-CacheAware-Parallel`

2. If matrix sizes are frequently 2048+:
   - Prefer CUDA path
   - Keep a warm-up call to reduce first-iteration penalty in measured workload

3. Keep `Naive-ijkLoop` only as a correctness/performance baseline reference.

## Evidence Sources

- `logs_1/matmul_bench_64_all_20260213_012711_715404.log`
- `logs_1/matmul_bench_128_all_20260213_012711_880223.log`
- `logs_1/matmul_bench_256_all_20260213_012712_044429.log`
- `logs_1/matmul_bench_512_all_20260213_012712_322203.log`
- `logs_1/matmul_bench_1024_all_20260213_012713_492694.log`
- `logs_1/matmul_bench_2048_all_20260213_012724_280765.log`
- `logs_1/matmul_bench_3072_all_20260213_012927_501673.log`
- `logs_1/matmul_bench_4096_all_20260213_013639_517027.log`
