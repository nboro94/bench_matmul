#include "benchmark.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

std::vector<BenchmarkResult> benchmarkResults;
double GLOBAL_BASELINE_DURATION = -1.0;
std::string GLOBAL_BASELINE_NAME = "";
bool GLOBAL_BENCHMARK_KERNEL_FAILURE = false;
std::string GLOBAL_BENCHMARK_FAILURE_REASON = "";

void benchmarkMultiplication(float **matrixA, float **matrixB, float **result,
                             void (*multiplyFunc)(float **, float **, float **),
                             const std::string &methodName) {
  log("Beginning benchmark of " + methodName);

  // Run multiple iterations to get more stable measurements
  const int iterations = 3;
  double totalDuration = 0.0;

  // Execute and time the multiplication algorithm multiple times
  for (int iter = 0; iter < iterations; iter++) {
    log("  Running iteration " + std::to_string(iter + 1) + " of " +
        std::to_string(iterations));

    // Measure execution time with high precision
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the multiplication function
    GLOBAL_BENCHMARK_KERNEL_FAILURE = false;
    GLOBAL_BENCHMARK_FAILURE_REASON.clear();
    multiplyFunc(matrixA, matrixB, result);
    if (GLOBAL_BENCHMARK_KERNEL_FAILURE) {
      throw std::runtime_error("Method '" + methodName +
                               "' failed: " + GLOBAL_BENCHMARK_FAILURE_REASON);
    }

    // Calculate elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double duration_ms = duration_us.count() / 1000.0;

    log("  Iteration " + std::to_string(iter + 1) + " completed in " +
        std::to_string(duration_ms) + " ms");

    totalDuration += duration_ms;
  }

  // Calculate average execution time
  double avgDuration = totalDuration / iterations;
  log("Average execution time for " + methodName + ": " +
      std::to_string(avgDuration) + " ms");

  // Calculate performance relative to baseline implementation
  // If `Naive-ijkLoop` is executed it will become the baseline. Otherwise
  // the first executed method will be used as the baseline so that
  // comparisons still make sense when the user intentionally omits naive.
  double speedup = 1.0;
  if (GLOBAL_BASELINE_DURATION < 0.0) {
    GLOBAL_BASELINE_DURATION = avgDuration;
    GLOBAL_BASELINE_NAME = methodName;
    speedup = 1.0;
  } else {
    speedup = GLOBAL_BASELINE_DURATION / avgDuration;
  }

  // Store results for final comparison table
  BenchmarkResult benchmarkResult;
  benchmarkResult.methodName = methodName;
  benchmarkResult.averageDuration = avgDuration;
  benchmarkResult.speedup = speedup;
  benchmarkResults.push_back(benchmarkResult);

  log("Benchmark of " + methodName + " completed");
}

void displayPerformanceComparisonTable() {
  log("Generating performance comparison table");

  // Format and print table header
  std::cout << "\n-------------------------------------------------------------"
               "----------------"
            << std::endl;
  std::cout << "PERFORMANCE COMPARISON TABLE" << std::endl;
  std::cout << "---------------------------------------------------------------"
               "----------------"
            << std::endl;
  std::cout << std::left << std::setw(55) << "Implementation" << std::right
            << std::setw(14) << "Time (ms)" << std::right << std::setw(20)
            << "Speedup" << std::endl;
  std::cout << "---------------------------------------------------------------"
               "----------------"
            << std::endl;

  // Sort results by execution time (fastest first)
  std::vector<BenchmarkResult> sortedResults = benchmarkResults;
  std::sort(sortedResults.begin(), sortedResults.end(),
            [](const BenchmarkResult &a, const BenchmarkResult &b) {
              return a.averageDuration < b.averageDuration;
            });

  // Print all results with formatting
  for (const auto &result : sortedResults) {
    std::cout << std::left << std::setw(55) << result.methodName << std::right
              << std::fixed << std::setprecision(3) << std::setw(14)
              << result.averageDuration;

    if (!GLOBAL_BASELINE_NAME.empty() &&
        result.methodName == GLOBAL_BASELINE_NAME) {
      std::cout << std::right << std::setw(20) << "baseline";
    } else {
      std::cout << std::right << std::setw(20)
                << (std::to_string(result.speedup) + "x faster");
    }
    std::cout << std::endl;
  }

  // Find and report the fastest implementation
  double bestSpeedup = 0.0;
  std::string fastestMethod;
  for (const auto &result : benchmarkResults) {
    if (result.speedup > bestSpeedup) {
      bestSpeedup = result.speedup;
      fastestMethod = result.methodName;
    }
  }

  std::cout << "---------------------------------------------------------"
            << std::endl;
  std::cout << "Fastest implementation: " << fastestMethod << " (" << std::fixed
            << std::setprecision(2) << bestSpeedup << "x faster than "
            << GLOBAL_BASELINE_NAME << ")" << std::endl;
  std::cout << "---------------------------------------------------------"
            << std::endl;
}
