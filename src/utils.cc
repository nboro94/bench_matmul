#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>


bool VERBOSE_LOGGING = true;
int N = 512;
int BLOCK_SIZE = 32;
unsigned int NUM_THREADS = 1;

// Utility function to conditionally output log messages based on verbosity
// setting
void log(const std::string &message) {
  if (VERBOSE_LOGGING) {
    std::cout << "[LOG] " << message << std::endl;
  }
}

/**
 * Determines the optimal number of threads based on hardware capabilities.
 * Adjusts the global NUM_THREADS variable accordingly.
 */
void determineOptimalThreadCount() {
  // Query hardware for available concurrent threads
  unsigned int hwThreads = std::thread::hardware_concurrency();

  // Set thread count with reasonable bounds
  // Min: 1 thread, Max: 48 threads or hardware limit, whichever is less
  NUM_THREADS = std::max(1u, std::min(hwThreads, 48u));

  log("Hardware detection: " + std::to_string(hwThreads) +
      " logical CPU cores available");
  if (NUM_THREADS != hwThreads) {
    log("Thread count adjusted to " + std::to_string(NUM_THREADS) +
        " for optimal performance");
  }

  std::cout << "Hardware supports " << hwThreads << " concurrent threads"
            << std::endl;
  std::cout << "Using " << NUM_THREADS << " threads for parallel operations"
            << std::endl;
}

/**
 * Calculates the optimal block size based on CPU cache characteristics.
 * The goal is to fit three blocks (A, B, result) within the cache for
 * minimal cache misses during blocked multiplication algorithms.
 */
void determineOptimalBlockSize() {
  int cacheSizeKB = 0;

  // Attempt to detect L1 cache size from Linux system information
  std::ifstream cacheFile("/sys/devices/system/cpu/cpu0/cache/index1/size");
  if (cacheFile.is_open()) {
    std::string sizeStr;
    cacheFile >> sizeStr;
    cacheSizeKB = std::stoi(sizeStr);
    cacheFile.close();
  } else {
    // If L1 cache detection failed, try L2 cache
    std::ifstream cacheFileL2("/sys/devices/system/cpu/cpu0/cache/index2/size");
    if (cacheFileL2.is_open()) {
      std::string sizeStr;
      cacheFileL2 >> sizeStr;
      cacheSizeKB = std::stoi(sizeStr);
      cacheFileL2.close();
    } else {
      // Fall back to a conservative default if detection fails
      cacheSizeKB = 256;
      log("Could not detect cache size, assuming 256KB L2 cache");
    }
  }

  // Convert kilobytes to bytes for calculations
  int cacheSizeBytes = cacheSizeKB * 1024;

  // Calculate how much memory we can use per matrix block
  // We need space for 3 blocks: A, B, and result portions
  int maxMatrixSizeBytes = cacheSizeBytes / 3;
  int maxBlockElements = maxMatrixSizeBytes / sizeof(float);
  int optimalBlockSize = static_cast<int>(std::sqrt(maxBlockElements));

  // Round down to power of 2 for optimal memory alignment and performance
  int powerOf2 = 1;
  while (powerOf2 * 2 <= optimalBlockSize) {
    powerOf2 *= 2;
  }

  // Set the global block size parameter
  BLOCK_SIZE = powerOf2;

  std::cout << "Cache size detected: " << cacheSizeKB << "KB" << std::endl;
  std::cout << "Optimal block size: " << BLOCK_SIZE << std::endl;
}

// Memory tracking counters for leak detection
int matrixAllocations = 0;   // Number of matrices allocated
int matrixDeallocations = 0; // Number of matrices deallocated

/**
 * Allocates a contiguous block of memory for an N×N matrix.
 *
 * Uses a layout where:
 * - A single contiguous block holds all N² elements
 * - An array of N pointers provides row-based access
 *
 * Returns a pointer to the matrix (float**).
 */
float **allocateMatrix() {
  try {
    log("Allocating matrix of size " + std::to_string(N) + "x" +
        std::to_string(N));

    // Allocate the primary data block (N×N elements)
    float *data = new float[N * N];

    // Allocate array of row pointers for 2D indexing
    float **matrix = new float *[N];

    // Configure row pointers to point to appropriate positions in data block
    for (int i = 0; i < N; i++) {
      matrix[i] = &data[i * N];
    }

    // Track allocation for memory leak detection
    matrixAllocations++;
    log("Matrix allocated successfully (total: " +
        std::to_string(matrixAllocations) + ")");

    return matrix;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * Deallocates a matrix previously allocated with allocateMatrix().
 * Handles the special memory layout where all elements are in a contiguous
 * block.
 *
 * @param matrix Pointer to the matrix to deallocate
 */
void deallocateMatrix(float **matrix) {
  if (matrix == nullptr) {
    log("Warning: Attempted to deallocate nullptr matrix");
    return; // Prevent null pointer access
  }

  // Free the contiguous data block (stored at first row pointer)
  delete[] matrix[0];

  // Free the array of row pointers
  delete[] matrix;

  // Track deallocation for memory leak detection
  matrixDeallocations++;
  log("Matrix deallocated successfully (total: " +
      std::to_string(matrixDeallocations) + ")");
}

/**
 * Checks for memory leaks by comparing allocation and deallocation counts.
 * Logs detailed information about any detected leaks.
 */
void checkMemoryLeaks() {
  if (matrixAllocations == matrixDeallocations) {
    log("Memory check passed: All " + std::to_string(matrixAllocations) +
        " matrices were properly deallocated");
  } else {
    std::cerr << "WARNING: Memory leak detected!" << std::endl;
    std::cerr << "  Matrices allocated: " << matrixAllocations << std::endl;
    std::cerr << "  Matrices deallocated: " << matrixDeallocations << std::endl;
    std::cerr << "  Difference: " << matrixAllocations - matrixDeallocations
              << std::endl;
  }
}

/**
 * Initializes a matrix with random float values.
 * Standard implementation using nested loops.
 *
 * @param matrix The matrix to initialize
 * @param gen Random number generator
 * @param dist Distribution to generate values
 */
void initializeRandomMatrix(float **matrix, std::mt19937 &gen,
                            std::uniform_real_distribution<float> &dist) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrix[i][j] = dist(gen);
    }
  }
}

/**
 * Initializes a matrix with random float values using cache-optimized approach.
 * Takes advantage of contiguous memory layout for better cache utilization.
 *
 * @param matrix The matrix to initialize
 * @param gen Random number generator
 * @param dist Distribution to generate values
 */
void initializeRandomMatrixFast(float **matrix, std::mt19937 &gen,
                                std::uniform_real_distribution<float> &dist) {
  // Direct access to contiguous memory block
  float *data = matrix[0];

  // Linear traversal of memory for optimal cache performance
  for (int i = 0; i < N * N; i++) {
    data[i] = dist(gen);
  }
}

/**
 * Prints the contents of a matrix to standard output.
 *
 * @param matrix The matrix to print
 * @param name A descriptive name for the matrix
 */
void printMatrix(float **matrix, const std::string &name) {
  std::cout << name << ":\n";
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << matrix[i][j] << "\t";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

/**
 * Transposes a matrix using block-based algorithm for cache efficiency.
 *
 * @param input The source matrix to transpose
 * @param output The destination for the transposed result
 */
void transposeMatrix(float **input, float **output) {
  // Process matrix in blocks to improve cache locality
  for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
    for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
      // Process each element in the current block
      for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
        for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
          // Swap indices to perform transpose
          output[j][i] = input[i][j];
        }
      }
    }
  }
}

/**
 * Performs matrix transposition in parallel using multiple threads.
 * Each thread handles a subset of the matrix blocks.
 *
 * @param input The source matrix to transpose
 * @param output The destination for the transposed result
 */
void transposeMatrixParallel(float **input, float **output) {
  // Define the workload for each thread as a lambda function
  auto threadFunction = [&](int startBlock, int endBlock) {
    // Process assigned range of blocks
    for (int bi = startBlock; bi < endBlock; bi += BLOCK_SIZE) {
      // For each row block, process all column blocks
      for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
        // Transpose elements within the block
        for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
          for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
            output[j][i] = input[i][j];
          }
        }
      }
    }
  };

  // Create and launch worker threads
  std::vector<std::thread> threads;
  // Calculate block distribution among threads
  int blocksPerThread = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / NUM_THREADS;
  blocksPerThread = std::max(1, blocksPerThread) * BLOCK_SIZE;

  // Launch threads with appropriate workload assignments
  for (unsigned int t = 0; t < NUM_THREADS; t++) {
    int startBlock = t * blocksPerThread;
    // Ensure last thread handles any remaining blocks
    int endBlock = (t == NUM_THREADS - 1) ? N : (t + 1) * blocksPerThread;
    threads.push_back(std::thread(threadFunction, startBlock, endBlock));
  }

  // Wait for all threads to complete their work
  for (auto &thread : threads) {
    thread.join();
  }
}

/**
 * Compares two floating-point values for approximate equality.
 * Handles both absolute and relative differences.
 *
 * @param a First value to compare
 * @param b Second value to compare
 * @return True if values are approximately equal
 */
bool almostEqual(float a, float b) {
  const float absoluteEpsilon = 1e-6f; // For small absolute differences
  const float relativeEpsilon = 1e-5f; // For relative differences

  // Handle exact equality and very small numbers
  if (a == b ||
      (std::fabs(a) < absoluteEpsilon && std::fabs(b) < absoluteEpsilon)) {
    return true;
  }

  // Check relative error for larger numbers
  return std::fabs(a - b) <=
         relativeEpsilon * std::max(std::fabs(a), std::fabs(b));
}
