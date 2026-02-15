#include "benchmark.h"
#include "matmul.h"
#include "utils.h"
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

int main(int argc, char *argv[]) {
  try {
    log("Matrix multiplication benchmark program starting");

    // Map all methods
    std::vector<std::pair<std::string, void (*)(float **, float **, float **)>>
        allMethods = {
            {"Naive-ijkLoop", multiplyMatrices},
            {"Naive-ijkLoop-Parallel", multiplyMatricesNaiveParallel},
            {"BlockTiled-CacheAware", multiplyMatricesOptimized},
            {"BlockTiled-CacheAware-Parallel",
             multiplyMatricesOptimizedParallel},
            {"SIMD-AVX2-Transposed", multiplyMatricesAVX2},
            {"RowColumn-Transposed", multiplyMatricesTransposed},
            {"Scalar-LoopUnrolled", multiplyMatricesOptimizedNoSIMD},
            {"Parallel-SIMD-AVX2", multiplyMatricesThreaded},
            {"Parallel-Scalar-LoopUnrolled",
             multiplyMatricesOptimizedNoSIMDThreaded},
            {"SIMD-AVX2-Direct", multiplyMatricesAVX2NoTranspose},
            {"Parallel-SIMD-Direct", multiplyMatricesThreadedAVX2NoTranspose},
            {"BlockLocal-StackTranspose", multiplyMatricesLocalTranspose}
#ifdef HAVE_CUDA
            ,
            {"CUDA-Naive", multiplyMatricesCUDA}
#endif
#ifdef HAVE_PTHREAD
            ,
            {"Parallel-SIMD-Pthread", multiplyMatricesPthreadWrapper}
#endif
#ifdef HAVE_TBB
            ,
            {"Parallel-SIMD-TBB", multiplyMatricesTBBWrapper}
#endif
        };

    // Process command line arguments
    std::set<std::string> methodsToRun;
    bool listOnly = false;
    bool threadsSetByUser = false;
    std::string baselineRequested = "";
    if (argc > 1) {
      for (int argi = 1; argi < argc; ++argi) {
        std::string a = argv[argi];
        if (a == "--list") {
          listOnly = true;
        } else if (a.rfind("--threads=", 0) == 0) {
          std::string payload = a.substr(10);
          try {
            int parsedThreads = std::stoi(payload);
            if (parsedThreads > 0) {
              NUM_THREADS = parsedThreads;
              threadsSetByUser = true;
            }
          } catch (const std::exception &e) {
            // ignore
          }
        } else if (a == "--threads" && argi + 1 < argc) {
          std::string payload = argv[++argi];
          try {
            int parsedThreads = std::stoi(payload);
            if (parsedThreads > 0) {
              NUM_THREADS = parsedThreads;
              threadsSetByUser = true;
            }
          } catch (const std::exception &e) {
            // ignore
          }
        } else if (a.rfind("--run=", 0) == 0) {
          std::string payload = a.substr(6);
          std::stringstream ss(payload);
          std::string token;
          while (std::getline(ss, token, ',')) {
            if (!token.empty())
              methodsToRun.insert(token);
          }
        } else if (a.rfind("--baseline=", 0) == 0) {
          baselineRequested = a.substr(11);
        } else if (a == "--baseline" && argi + 1 < argc) {
          baselineRequested = argv[++argi];
        } else if (a == "--run" && argi + 1 < argc) {
          std::string payload = argv[++argi];
          std::stringstream ss(payload);
          std::string token;
          while (std::getline(ss, token, ',')) {
            if (!token.empty())
              methodsToRun.insert(token);
          }
        } else {
          bool looksLikeNumber = !a.empty();
          for (char c : a) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
              looksLikeNumber = false;
              break;
            }
          }
          if (looksLikeNumber) {
            try {
              int parsedN = std::stoi(a);
              if (parsedN <= 0) {
                std::cerr << "ERROR: Matrix dimension must be positive. Using "
                             "default N = 512"
                          << std::endl;
              } else {
                N = parsedN;
                std::cout << "Using matrix dimension N = " << N
                          << " from command line argument" << std::endl;
              }
            } catch (const std::exception &e) {
              std::cerr << "ERROR: Invalid matrix dimension: '" << a
                        << "'. Using default N = 512" << std::endl;
              N = 512;
            }
          } else {
            log("Ignoring unrecognized command-line token: " + a);
          }
        }
      }
    } else {
      std::cout << "No command-line arguments detected.\n\n";
      std::cout << "Usage:\n";
      std::cout
          << "  [N]                : set matrix dimension (positive integer)\n";
      std::cout << "  --threads [T]      : set number of threads (positive "
                   "integer)\n";
      std::cout
          << "  --list             : list available multiplication methods\n";
      std::cout << "  --run=name1,name2  : run only the named methods "
                   "(comma-separated)\n\n";
      std::cout << "Available multiplication methods:\n";
      for (const auto &m : allMethods) {
        std::cout << "  " << m.first << "\n";
      }
      std::cout << "Example:\n";
      std::cout << "  ./bench-matmul 1024 "
                   "--run=BlockTiled-CacheAware,SIMD-AVX2-Transposed\n\n";
      return 0;
    }

    if (listOnly) {
      std::cout << "Available multiplication methods:\n";
      for (const auto &m : allMethods) {
        std::cout << "  " << m.first << "\n";
      }
      return 0;
    }

    log("Matrix dimension set to N = " + std::to_string(N));

    bool isPowerOf2 = (N & (N - 1)) == 0;
    if (!isPowerOf2) {
      log("NOTE: N = " + std::to_string(N) +
          " is not a power of 2. Some optimizations may be less effective.");
    }

    log("Determining optimal block size based on CPU cache");
    determineOptimalBlockSize();

    if (BLOCK_SIZE > N / 2) {
      BLOCK_SIZE = std::max(8, N / 4);
      log("Block size adjusted to " + std::to_string(BLOCK_SIZE) +
          " to better fit matrix dimensions");
    }

    if (!threadsSetByUser) {
      log("Determining optimal thread count based on CPU cores");
      determineOptimalThreadCount();
    } else {
      log("Thread count set to " + std::to_string(NUM_THREADS) + " by user");
      std::cout << "Using " << NUM_THREADS
                << " threads for parallel operations (set by user)"
                << std::endl;
    }

    std::cout << "Matrix Multiplication with Random Matrices (N=" << N
              << ", BLOCK_SIZE=" << BLOCK_SIZE << ")\n\n";

    log("Setting up random number generation");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);

    auto shouldRun = [&](const std::string &name) -> bool {
      return methodsToRun.empty() ||
             methodsToRun.find(name) != methodsToRun.end();
    };

    std::unordered_set<std::string> knownMethods;
    for (const auto &m : allMethods) {
      knownMethods.insert(m.first);
    }
    for (const auto &requested : methodsToRun) {
      if (knownMethods.find(requested) == knownMethods.end()) {
        std::cerr << "ERROR: Requested method '" << requested
                  << "' is not a known method.\n";
        return 1;
      }
    }

    std::string baselineName = "";
    if (!baselineRequested.empty()) {
      bool found = false;
      for (const auto &m : allMethods)
        if (m.first == baselineRequested) {
          found = true;
          break;
        }
      if (!found) {
        std::cerr << "ERROR: Requested baseline '" << baselineRequested
                  << "' is not a known method.\n";
        return 1;
      }
      if (methodsToRun.empty() ||
          methodsToRun.find(baselineRequested) == methodsToRun.end()) {
        methodsToRun.insert(baselineRequested);
      }
      baselineName = baselineRequested;
    } else if (shouldRun("Naive-ijkLoop")) {
      baselineName = "Naive-ijkLoop";
    } else {
      baselineName = "";
    }

    std::vector<std::pair<std::string, void (*)(float **, float **, float **)>>
        execOrder;
    if (!baselineName.empty()) {
      for (const auto &m : allMethods) {
        if (m.first == baselineName) {
          execOrder.push_back(m);
          break;
        }
      }
    }
    for (const auto &m : allMethods) {
      if (!shouldRun(m.first))
        continue;
      if (!baselineName.empty() && m.first == baselineName)
        continue;
      execOrder.push_back(m);
    }

    if (!baselineName.empty())
      GLOBAL_BASELINE_NAME = baselineName;

    struct MatrixPool {
      std::vector<float **> matrices;

      float **allocateTracked() {
        float **m = allocateMatrix();
        matrices.push_back(m);
        return m;
      }

      void cleanup() {
        for (float **m : matrices) {
          deallocateMatrix(m);
        }
        matrices.clear();
      }

      ~MatrixPool() { cleanup(); }
    } matrixPool;

    log("Allocating matrices");
    float **aMatrix = matrixPool.allocateTracked();
    float **bMatrix = matrixPool.allocateTracked();

    // Allocate product buffers for each method
    std::unordered_map<std::string, float **> productMap;
    for (const auto &m : allMethods) {
      productMap[m.first] = matrixPool.allocateTracked();
    }

    log("Initializing matrices with random values");
    initializeRandomMatrixFast(aMatrix, gen, dist);
    initializeRandomMatrixFast(bMatrix, gen, dist);

    log("Beginning benchmark sequence");
    for (const auto &entry : execOrder) {
      const std::string &name = entry.first;
      auto func = entry.second;
      float **outBuf = productMap[name];
      benchmarkMultiplication(aMatrix, bMatrix, outBuf, func, name);
    }

    log("Verifying correctness of results from selected multiplication "
        "methods");

    // There might be no baseline if user ran only one method (e.g. without
    // --baseline arg and without Naive-ijkLoop) But benchmarkMultiplication
    // sets GLOBAL_BASELINE_NAME to the first executed method if not set.
    if (!GLOBAL_BASELINE_NAME.empty()) {
      float **baselineBuf = productMap[GLOBAL_BASELINE_NAME];

      std::vector<std::string> runMethods;
      for (const auto &entry : allMethods) {
        if (shouldRun(entry.first) && entry.first != GLOBAL_BASELINE_NAME) {
          runMethods.push_back(entry.first);
        }
      }

      for (int i = 0; i < N; i++) {
        if (i % 1000 == 0) {
          log("Verification progress: checking row " + std::to_string(i) +
              " of " + std::to_string(N));
        }
        for (int j = 0; j < N; j++) {
          for (const auto &mname : runMethods) {
            float **cmpBuf = productMap[mname];
            if (!almostEqual(baselineBuf[i][j], cmpBuf[i][j])) {
              log("ALERT: Discrepancy found at position [" + std::to_string(i) +
                  "][" + std::to_string(j) + "] between baseline and " + mname);
              std::cout << "Discrepancy at [" << i << "][" << j
                        << "]: baseline=" << baselineBuf[i][j] << "  " << mname
                        << "=" << cmpBuf[i][j] << std::endl;
              throw std::runtime_error(
                  "Result mismatch detected at position [" + std::to_string(i) +
                  "][" + std::to_string(j) + "]");
            }
          }
        }
      }
      log("Verification successful: All implementations produced identical "
          "results");
    }

    log("Generating performance summary");
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "PERFORMANCE SUMMARY" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;
    std::cout << "CPU threads: " << NUM_THREADS << " (of "
              << std::thread::hardware_concurrency() << " available)"
              << std::endl;

    double memoryUsageKB = (matrixAllocations * sizeof(float) * N * N) / 1024.0;
    double memoryUsageMB = memoryUsageKB / 1024.0;
    std::cout << "Total memory usage: " << std::fixed << std::setprecision(2);
    if (memoryUsageMB >= 1.0) {
      std::cout << memoryUsageMB << " MB" << std::endl;
    } else {
      std::cout << memoryUsageKB << " KB" << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;

    displayPerformanceComparisonTable();

    log("Deallocating matrices");
    matrixPool.cleanup();

    log("Checking for memory leaks");
    checkMemoryLeaks();

    log("Program completed successfully");
    return 0;
  } catch (const std::exception &e) {
    log("EXCEPTION: " + std::string(e.what()));
    std::cerr << "Exception occurred: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    log("EXCEPTION: Unknown exception occurred");
    std::cerr << "Unknown exception occurred" << std::endl;
    return 1;
  }
}
