#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>

/**
 * Structure to store benchmark results for comparison.
 */
struct BenchmarkResult {
  std::string methodName; // Name of multiplication method
  double averageDuration; // Average execution time in milliseconds
  double speedup;         // Performance ratio relative to baseline
};

// Collection of all benchmark results for final comparison
extern std::vector<BenchmarkResult> benchmarkResults;

// Global baseline tracking: stores the name and duration of the chosen baseline
extern double GLOBAL_BASELINE_DURATION;
extern std::string GLOBAL_BASELINE_NAME;
extern bool GLOBAL_BENCHMARK_KERNEL_FAILURE;
extern std::string GLOBAL_BENCHMARK_FAILURE_REASON;

/**
 * Benchmarks a matrix multiplication function and records performance metrics.
 *
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix for storing results
 * @param multiplyFunc Function pointer to the multiplication algorithm to
 * benchmark
 * @param methodName Descriptive name of the algorithm for reporting
 */
void benchmarkMultiplication(float **matrixA, float **matrixB, float **result,
                             void (*multiplyFunc)(float **, float **, float **),
                             const std::string &methodName);

/**
 * Generates and displays a formatted performance comparison table.
 * Shows all benchmarked algorithms sorted by speed.
 */
void displayPerformanceComparisonTable();

#endif // BENCHMARK_H
