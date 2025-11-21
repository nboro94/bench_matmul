
# Benchmark Conclusions

##Scope

This report summarizes empirical performance measurements for four matrix-multiplication implementations measured on the current environment. Implementations compared:

- BlockTiled-CacheAware-Parallel (`tiled-par`)
- BlockTiled-CacheAware (serial) (`tiled`)
- Naive-ijkLoop (serial) (`naive`)
- Naive-ijkLoop-Parallel (`naive-par`)

##Data

Raw timing data are available as CSV files under `build/logs_*`. Measurements span a range of matrix sizes; representative points used in this report include N ∈ {100, 500, 1000, 3000}.

##Key findings

- For small matrices (N ≲ 400) `naive-par` often gives the lowest runtime because its simple row-partitioning parallelism incurs low overhead.
- For larger matrices (N ≳ 500) `tiled-par` consistently achieves the best absolute performance. Blocking reduces memory traffic and increases cache reuse; combining blocking with parallelism yields the largest gains at scale.
- The serial blocked algorithm (`tiled`) substantially outperforms the serial naive implementation (`naive`), indicating that cache-aware blocking is an effective optimization even without threads.
- The serial naive implementation (`naive`) is the slowest and is appropriate primarily as a correctness baseline.

##Numeric summary (selected points)

| Size | tiled-par (ms) | tiled (ms) | naive (ms) | naive-par (ms) |
|------:|---------------:|-----------:|----------:|---------------:|
| 100  | 0.497          | 0.664      | 0.962     | 0.321          |
| 500  | 38.675         | 64.717     | 124.487   | 48.497         |
| 1000 | 313.674        | 508.216    | 932.109   | 394.979        |
| 3000 | 8170.526       | 13769.665  | 32248.761 | 13191.149      |

##Performance interpretation

- Observed speedups over serial naive are typically in the 2×–3× range for parallel implementations; gains are limited by memory bandwidth and cache behavior rather than pure arithmetic throughput.
- Blocking reduces data movement and therefore improves scaling for larger N; the crossover point where blocking+parallelism outperforms naive-par depends on CPU/cache characteristics but appears in the tested runs near N≈300–500.
- Timing at very small sizes (N ≲ 50) is noisy and occasionally shows artifacts (timer resolution, scheduling jitter). Such sizes should not be used to draw firm conclusions without increasing iteration counts.

##Recommendations

- Use `BlockTiled-CacheAware-Parallel` (`tiled-par`) for production or large problem sizes.
- For quick experiments and small-to-medium sizes, `Naive-ijkLoop-Parallel` (`naive-par`) is an efficient, low-overhead option.
- Retain `Naive-ijkLoop` as a correctness baseline only.

##Measurement best practices

- Run multiple independent trials and report mean ± standard deviation (the harness currently averages three iterations per method).
- Exclude or increase iterations for sizes that produce sub-millisecond runtimes to avoid measurement noise.

##Suggested next experiments

1. Thread-scaling: measure performance of parallel implementations while varying thread count to quantify parallel efficiency and identify the practical thread limit.
2. Block-size tuning: sweep `BLOCK_SIZE` for blocked implementations to identify the cache-optimal block size on the target machine.
3. Profiling: collect CPU and memory-bandwidth metrics (e.g., `perf`) during large runs to identify whether memory bandwidth is the primary bottleneck.
4. SIMD/compiler optimization: benchmark vectorized variants and compiler flags (for example, `-O3 -march=native`) to measure vectorization benefits.

##Reproducibility

Re-run the comparison with the project runner. Example command:

```bash
python3 run_benchmarks.py --executable ./build/bench-matmul \
  --compare=tiled-par,tiled,naive,naive-par \
  --sizes=100,200,300,400,500,600,700,800,900,1000 -j 1
```

##Generated artifacts

Each run creates a human-readable text file and a CSV file under `build/logs_*` (e.g., `comparison_<methods>_<timestamp>.txt` and `.csv`). Use the CSVs for plotting and further analysis.
