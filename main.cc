#include <iostream>     // Input/output streams
#include <algorithm>    // Standard algorithms (min, max, etc.)
#include <vector>       // Dynamic array container
#include <random>       // Random number generation facilities
#include <ctime>        // Time-related utilities
#include <chrono>       // High-precision timing utilities
#include <iomanip>      // Output formatting
#include <fstream>      // File input/output
#include <cmath>        // Mathematical functions
#include <string>       // String manipulation utilities
#include <immintrin.h>  // Intel SIMD intrinsics for AVX2 operations
#include <thread>       // Threading support
#include <cstring>      // C-style string operations (memset, etc.)
#include <unordered_map>
#include <set>
#include <sstream>

// Global logging control
// Set to true for detailed execution logs, false for minimal output
bool VERBOSE_LOGGING = true;

// Utility function to conditionally output log messages based on verbosity setting
void log(const std::string& message) {
    if (VERBOSE_LOGGING) {
        std::cout << "[LOG] " << message << std::endl;
    }
}

// Configuration parameters
// Default matrix dimension (N×N matrices will be created)
int N = 512;
// Default block size for tiled operations (tuned based on cache characteristics)
int BLOCK_SIZE = 32;

// Parallelization parameter
// Number of execution threads (will be set based on hardware capabilities)
unsigned int NUM_THREADS = 1;

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
    
    log("Hardware detection: " + std::to_string(hwThreads) + " logical CPU cores available");
    if (NUM_THREADS != hwThreads) {
        log("Thread count adjusted to " + std::to_string(NUM_THREADS) + " for optimal performance");
    }
    
    std::cout << "Hardware supports " << hwThreads << " concurrent threads" << std::endl;
    std::cout << "Using " << NUM_THREADS << " threads for parallel operations" << std::endl;
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
    int maxBlockElements = maxMatrixSizeBytes / sizeof(int);
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
int matrixAllocations = 0;    // Number of matrices allocated
int matrixDeallocations = 0;  // Number of matrices deallocated

/**
 * Allocates a contiguous block of memory for an N×N matrix.
 * 
 * Uses a layout where:
 * - A single contiguous block holds all N² elements
 * - An array of N pointers provides row-based access
 * 
 * Returns a pointer to the matrix (float**).
 */
float** allocateMatrix() {
    try {
        log("Allocating matrix of size " + std::to_string(N) + "x" + std::to_string(N));
        
        // Allocate the primary data block (N×N elements)
        float* data = new float[N * N];
        
        // Allocate array of row pointers for 2D indexing
        float** matrix = new float*[N];
        
        // Configure row pointers to point to appropriate positions in data block
        for (int i = 0; i < N; i++) {
            matrix[i] = &data[i * N];
        }
        
        // Track allocation for memory leak detection
        matrixAllocations++;
        log("Matrix allocated successfully (total: " + std::to_string(matrixAllocations) + ")");
        
        return matrix;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Transposes a matrix using block-based algorithm for cache efficiency.
 * 
 * @param input The source matrix to transpose
 * @param output The destination for the transposed result
 */
void transposeMatrix(float** input, float** output) {
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
void transposeMatrixParallel(float** input, float** output) {
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
    for (auto& thread : threads) {
        thread.join();
    }
}

/**
 * Deallocates a matrix previously allocated with allocateMatrix().
 * Handles the special memory layout where all elements are in a contiguous block.
 * 
 * @param matrix Pointer to the matrix to deallocate
 */
void deallocateMatrix(float** matrix) {
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
    log("Matrix deallocated successfully (total: " + std::to_string(matrixDeallocations) + ")");
}

/**
 * Checks for memory leaks by comparing allocation and deallocation counts.
 * Logs detailed information about any detected leaks.
 */
void checkMemoryLeaks() {
    if (matrixAllocations == matrixDeallocations) {
        log("Memory check passed: All " + std::to_string(matrixAllocations) + " matrices were properly deallocated");
    } else {
        std::cerr << "WARNING: Memory leak detected!" << std::endl;
        std::cerr << "  Matrices allocated: " << matrixAllocations << std::endl;
        std::cerr << "  Matrices deallocated: " << matrixDeallocations << std::endl;
        std::cerr << "  Difference: " << matrixAllocations - matrixDeallocations << std::endl;
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
void initializeRandomMatrix(float** matrix, std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
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
void initializeRandomMatrixFast(float** matrix, std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    // Direct access to contiguous memory block
    float* data = matrix[0];
    
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
void printMatrix(float** matrix, const std::string& name) {
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
 * Performs matrix multiplication using the standard nested triple-loop algorithm.
 * This serves as the baseline implementation for performance comparison.
 * Time complexity: O(N³)
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatrices(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));
    
    // Standard triple-loop matrix multiplication
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            for (int inner = 0; inner != N; inner++) {
                result[row][col] += matrixA[row][inner] * matrixB[inner][col];
            }
        }
    }
}

/**
 * Performs matrix multiplication using cache-aware blocking/tiling technique.
 * Divides matrices into blocks that fit in cache for improved performance.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesOptimized(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));
    
    // Process matrices in blocks for better cache utilization
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // For each block in result matrix, compute contribution
            // from all corresponding blocks in input matrices
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Process elements within current block
                for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                    for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                        // Cache A element to reduce memory access in inner loop
                        float aik = matrixA[i][k];
                        for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                            result[i][j] += aik * matrixB[k][j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * Parallel version of the cache-aware blocked/tiled multiplication.
 *
 * Partitions the block-rows among `NUM_THREADS` worker threads so each
 * thread works on disjoint sets of result block-rows. Inside each block
 * the computation mirrors `multiplyMatricesOptimized`.
 */
void multiplyMatricesOptimizedParallel(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));

    // Worker that processes a range of block-rows [startBlockRow, endBlockRow)
    auto worker = [&](int startBlockRow, int endBlockRow) {
        for (int bi = startBlockRow; bi < endBlockRow; bi += BLOCK_SIZE) {
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                    int iMax = std::min(bi + BLOCK_SIZE, N);
                    int jMax = std::min(bj + BLOCK_SIZE, N);
                    int kMax = std::min(bk + BLOCK_SIZE, N);

                    for (int i = bi; i < iMax; i++) {
                        for (int k = bk; k < kMax; k++) {
                            float aik = matrixA[i][k];
                            float* resRow = result[i];
                            float* brow = matrixB[k];
                            for (int j = bj; j < jMax; j++) {
                                resRow[j] += aik * brow[j];
                            }
                        }
                    }
                }
            }
        }
    };

    // Compute number of block-rows and distribute them among threads
    int totalBlockRows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocksPerThread = std::max(1, (totalBlockRows + (int)NUM_THREADS - 1) / (int)NUM_THREADS);

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (unsigned int t = 0; t < NUM_THREADS; ++t) {
        int startBlockIdx = t * blocksPerThread;
        int endBlockIdx = std::min(totalBlockRows, startBlockIdx + blocksPerThread);
        if (startBlockIdx >= endBlockIdx) break; // no more work

        int startBi = startBlockIdx * BLOCK_SIZE;
        int endBi = endBlockIdx * BLOCK_SIZE;
        if (startBi >= N) break;
        endBi = std::min(endBi, N);

        threads.emplace_back(worker, startBi, endBi);
    }

    for (auto &th : threads) th.join();
}

/**
 * Naive triple-loop matrix multiplication but parallelized over rows.
 * Each worker thread computes a disjoint range of output rows using the
 * straightforward i-j-k algorithm. This gives a simple, easy-to-reason
 * parallel baseline for comparison.
 */
void multiplyMatricesNaiveParallel(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));

    // Worker lambda: compute rows [startRow, endRow)
    auto worker = [&](int startRow, int endRow) {
        for (int row = startRow; row < endRow; row++) {
            for (int col = 0; col < N; col++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += matrixA[row][k] * matrixB[k][col];
                }
                result[row][col] = sum;
            }
        }
    };

    // Partition rows among threads
    unsigned int threads = std::max(1u, NUM_THREADS);
    int rowsPerThread = N / threads;
    std::vector<std::thread> threadPool;
    for (unsigned int t = 0; t < threads; ++t) {
        int start = t * rowsPerThread;
        int end = (t == threads - 1) ? N : (t + 1) * rowsPerThread;
        if (start >= end) break;
        threadPool.emplace_back(worker, start, end);
    }

    for (auto &th : threadPool) th.join();
}

/**
 * Matrix multiplication using on-the-fly local transposition.
 * Transposes small blocks of matrix B into stack-allocated buffer
 * to improve cache behavior without extra memory allocation.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesLocalTranspose(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));
    
    // Allocate aligned buffer on stack for block transposition
    alignas(32) float localTransposedB[BLOCK_SIZE][BLOCK_SIZE];
    
    // Process matrices in blocks
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // For each block in result matrix, process corresponding blocks in inputs
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Transpose current block of B into local buffer
                for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        localTransposedB[k-bk][j-bj] = matrixB[k][j];
                    }
                }
                
                // Process current block with transposed data
                for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        // Accumulate dot product for current element
                        float sum = result[i][j];
                        
                        // Access transposed data from local buffer for better cache behavior
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            sum += matrixA[i][k] * localTransposedB[k-bk][j-bj];
                        }
                        
                        result[i][j] = sum;
                    }
                }
            }
        }
    }
}

/**
 * Performs matrix multiplication with transposition and tiling.
 * Transposes matrix B first to improve memory access patterns.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesTransposed(float** matrixA, float** matrixB, float** result) {
    // Create transposed copy of B for better cache locality
    float** transposedB = allocateMatrix();
    transposeMatrix(matrixB, transposedB);
    
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));
    
    // Process matrices in blocks
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Process current block
                for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        // Accumulate dot product for this element
                        float sum = 0;
                        // With transposition, both matrices are accessed in row-major order
                        // for better cache efficiency
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            sum += matrixA[i][k] * transposedB[j][k];
                        }
                        result[i][j] += sum;
                    }
                }
            }
        }
    }
    
    // Release temporary transposed matrix
    deallocateMatrix(transposedB);
}

/**
 * Performs SIMD-optimized matrix multiplication using AVX2 vector instructions.
 * Uses transposition and processes 8 elements at once with 256-bit vectors.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesAVX2(float** matrixA, float** matrixB, float** result) {
    // Create transposed copy of B for better memory access patterns
    float** transposedB = allocateMatrix();
    transposeMatrix(matrixB, transposedB);
    
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));

    // Process in blocks for cache efficiency
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // Process elements in current block
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                // Process 8 columns at once using AVX2 SIMD instructions
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize 256-bit accumulator register to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate products for all corresponding elements
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcast single A value to all 8 lanes (SIMD replication)
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from transposed B
                            __m256 b_vals = _mm256_setr_ps(
                                transposedB[j][k], transposedB[j+1][k], 
                                transposedB[j+2][k], transposedB[j+3][k],
                                transposedB[j+4][k], transposedB[j+5][k], 
                                transposedB[j+6][k], transposedB[j+7][k]
                            );
                            
                            // Fused multiply-add: c_vals += a_val * b_vals
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store computed values back to result matrix
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining columns (when N not divisible by 8)
                for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8); j < std::min(bj + BLOCK_SIZE, N); j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += matrixA[i][k] * transposedB[j][k];
                    }
                    result[i][j] = sum;
                }
            }
        }
    }
    
    // Release temporary transposed matrix
    deallocateMatrix(transposedB);
}

/**
 * SIMD-optimized matrix multiplication without pre-transposition.
 * Uses AVX2 intrinsics but accesses matrix B directly, which can be
 * less cache-efficient but saves the transposition cost.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesAVX2NoTranspose(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));

    // Process in blocks for cache efficiency
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // Process elements in current block
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                // Process 8 columns at once using AVX2 SIMD instructions
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize 256-bit accumulator register to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate products for all corresponding elements
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcast single A value to all 8 lanes
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from B directly (less cache-friendly)
                            __m256 b_vals = _mm256_setr_ps(
                                matrixB[k][j], matrixB[k][j+1], 
                                matrixB[k][j+2], matrixB[k][j+3],
                                matrixB[k][j+4], matrixB[k][j+5], 
                                matrixB[k][j+6], matrixB[k][j+7]
                            );
                            
                            // Fused multiply-add: c_vals += a_val * b_vals
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store computed values back to result matrix
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining columns (when N not divisible by 8)
                for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8); j < std::min(bj + BLOCK_SIZE, N); j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += matrixA[i][k] * matrixB[k][j];
                    }
                    result[i][j] = sum;
                }
            }
        }
    }
}

/**
 * Multithreaded matrix multiplication with AVX2 SIMD instructions.
 * Combines parallel execution with SIMD vectorization for maximum performance.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesThreaded(float** matrixA, float** matrixB, float** result) {
    // Create transposed copy of B for better memory access patterns
    // Use parallel transposition for better performance
    float** transposedB = allocateMatrix();
    transposeMatrixParallel(matrixB, transposedB);
    
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));

    // Define work function for each thread
    auto threadFunction = [&](int startRow, int endRow, int threadId) {
        // Each thread processes its assigned rows
        for (int i = startRow; i < endRow; i++) {
            // Process in blocks for cache efficiency
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                // Process 8 columns at once using AVX2 SIMD instructions
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize 256-bit accumulator register to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate products for all corresponding elements
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcast single A value to all 8 lanes
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from transposed B
                            __m256 b_vals = _mm256_setr_ps(
                                transposedB[j][k], transposedB[j+1][k], 
                                transposedB[j+2][k], transposedB[j+3][k],
                                transposedB[j+4][k], transposedB[j+5][k], 
                                transposedB[j+6][k], transposedB[j+7][k]
                            );
                            
                            // Fused multiply-add: c_vals += a_val * b_vals
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store computed values back to result matrix
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining columns (when N not divisible by 8)
                for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8); j < std::min(bj + BLOCK_SIZE, N); j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += matrixA[i][k] * transposedB[j][k];
                    }
                    result[i][j] = sum;
                }
            }
        }
    };
    
    // Create and launch worker threads
    std::vector<std::thread> threads;
    int rowsPerThread = N / NUM_THREADS;
    
    // Distribute rows among threads
    for (unsigned int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        // Ensure last thread handles any remaining rows
        int endRow = (t == NUM_THREADS - 1) ? N : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(threadFunction, startRow, endRow, t));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Release temporary transposed matrix
    deallocateMatrix(transposedB);
}

/**
 * Optimized matrix multiplication without SIMD instructions.
 * Uses transposition, tiling, and manual loop unrolling for performance.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesOptimizedNoSIMD(float** matrixA, float** matrixB, float** result) {
    // Create transposed copy of B for better memory access patterns
    float** transposedB = allocateMatrix();
    transposeMatrix(matrixB, transposedB);
    
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));
    
    // Process matrices in blocks for cache efficiency
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Process current block with optimized loop ordering
                for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        // Use scalar register for accumulation
                        float sum = result[i][j];
                        
                        // Manual loop unrolling (process 4 elements at once)
                        // This enables better instruction pipelining
                        int k = bk;
                        for (; k < std::min(bk + BLOCK_SIZE, N) - 3; k += 4) {
                            // Four operations unrolled
                            sum += matrixA[i][k] * transposedB[j][k];
                            sum += matrixA[i][k+1] * transposedB[j][k+1];
                            sum += matrixA[i][k+2] * transposedB[j][k+2];
                            sum += matrixA[i][k+3] * transposedB[j][k+3];
                        }
                        
                        // Handle remaining elements (when not divisible by 4)
                        for (; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            sum += matrixA[i][k] * transposedB[j][k];
                        }
                        
                        // Store accumulated result
                        result[i][j] = sum;
                    }
                }
            }
        }
    }
    
    // Release temporary transposed matrix
    deallocateMatrix(transposedB);
}

/**
 * Multithreaded optimized matrix multiplication without SIMD instructions.
 * Combines parallel execution with scalar optimizations.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesOptimizedNoSIMDThreaded(float** matrixA, float** matrixB, float** result) {
    // Create transposed copy of B for better memory access patterns
    float** transposedB = allocateMatrix();
    transposeMatrixParallel(matrixB, transposedB);
    
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));

    // Define work function for each thread
    auto threadFunction = [&](int startRow, int endRow, int threadId) {
        // Each thread processes its assigned rows
        for (int i = startRow; i < endRow; i++) {
            // Process in blocks for cache efficiency
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                    // Process block with optimized loop ordering
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        // Use scalar register for accumulation
                        float sum = result[i][j];
                        
                        // Manual loop unrolling (process 4 elements at once)
                        int k = bk;
                        for (; k < std::min(bk + BLOCK_SIZE, N) - 3; k += 4) {
                            // Four operations unrolled
                            sum += matrixA[i][k] * transposedB[j][k];
                            sum += matrixA[i][k+1] * transposedB[j][k+1];
                            sum += matrixA[i][k+2] * transposedB[j][k+2];
                            sum += matrixA[i][k+3] * transposedB[j][k+3];
                        }
                        
                        // Handle remaining elements (when not divisible by 4)
                        for (; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            sum += matrixA[i][k] * transposedB[j][k];
                        }
                        
                        // Store accumulated result
                        result[i][j] = sum;
                    }
                }
            }
        }
    };
    
    // Create and launch worker threads
    std::vector<std::thread> threads;
    int rowsPerThread = N / NUM_THREADS;
    
    // Distribute rows among threads
    for (unsigned int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? N : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(threadFunction, startRow, endRow, t));
    }
    
    // Wait for all threads to complete
    for (unsigned int t = 0; t < threads.size(); t++) {
        threads[t].join();
    }
    
    // Release temporary transposed matrix
    deallocateMatrix(transposedB);
}

/**
 * Multithreaded matrix multiplication with AVX2 without pre-transposition.
 * Combines parallel execution with SIMD vectorization but avoids the transposition cost.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 */
void multiplyMatricesThreadedAVX2NoTranspose(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros
    memset(result[0], 0, N * N * sizeof(float));

    // Define work function for each thread
    auto threadFunction = [&](int startRow, int endRow, int threadId) {
        // Each thread processes its assigned rows
        for (int i = startRow; i < endRow; i++) {
            // Process in blocks for cache efficiency
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                // Process 8 columns at once using AVX2 SIMD instructions
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize 256-bit accumulator register to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate products for all corresponding elements
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcast single A value to all 8 lanes
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from B directly (less cache-friendly)
                            __m256 b_vals = _mm256_setr_ps(
                                matrixB[k][j], matrixB[k][j+1], 
                                matrixB[k][j+2], matrixB[k][j+3],
                                matrixB[k][j+4], matrixB[k][j+5], 
                                matrixB[k][j+6], matrixB[k][j+7]
                            );
                            
                            // Fused multiply-add: c_vals += a_val * b_vals
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store computed values back to result matrix
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining columns (when N not divisible by 8)
                for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8); j < std::min(bj + BLOCK_SIZE, N); j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += matrixA[i][k] * matrixB[k][j];
                    }
                    result[i][j] = sum;
                }
            }
        }
    };
    
    // Create and launch worker threads
    std::vector<std::thread> threads;
    int rowsPerThread = N / NUM_THREADS;
    
    // Distribute rows among threads
    for (unsigned int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? N : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(threadFunction, startRow, endRow, t));
    }
    
    // Wait for all threads to complete
    for (unsigned int t = 0; t < threads.size(); t++) {
        threads[t].join();
    }
}


/**
 * Structure to store benchmark results for comparison.
 */
struct BenchmarkResult {
    std::string methodName;     // Name of multiplication method
    double averageDuration;     // Average execution time in milliseconds
    double speedup;             // Performance ratio relative to baseline
};

// Collection of all benchmark results for final comparison
std::vector<BenchmarkResult> benchmarkResults;

// Global baseline tracking: stores the name and duration of the chosen baseline
double GLOBAL_BASELINE_DURATION = -1.0;
std::string GLOBAL_BASELINE_NAME = "";

/**
 * Benchmarks a matrix multiplication function and records performance metrics.
 * 
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix for storing results
 * @param multiplyFunc Function pointer to the multiplication algorithm to benchmark
 * @param methodName Descriptive name of the algorithm for reporting
 */
void benchmarkMultiplication(float** matrixA, float** matrixB, float** result, 
                            void (*multiplyFunc)(float**, float**, float**),
                            const std::string& methodName) {
    log("Beginning benchmark of " + methodName);
    
    // Run multiple iterations to get more stable measurements
    const int iterations = 3;
    double totalDuration = 0.0;
    
    // Execute and time the multiplication algorithm multiple times
    for (int iter = 0; iter < iterations; iter++) {
        log("  Running iteration " + std::to_string(iter+1) + " of " + std::to_string(iterations));
        
        // Measure execution time with high precision
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute the multiplication function
        multiplyFunc(matrixA, matrixB, result);
        
        // Calculate elapsed time
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double duration_ms = duration_us.count() / 1000.0;
        
        log("  Iteration " + std::to_string(iter+1) + " completed in " + 
            std::to_string(duration_ms) + " ms");
        
        totalDuration += duration_ms;
    }
    
    // Calculate average execution time
    double avgDuration = totalDuration / iterations;
    log("Average execution time for " + methodName + ": " + std::to_string(avgDuration) + " ms");

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

/**
 * Generates and displays a formatted performance comparison table.
 * Shows all benchmarked algorithms sorted by speed.
 */
void displayPerformanceComparisonTable() {
    log("Generating performance comparison table");
    
    // Format and print table header
    std::cout << "\n-----------------------------------------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE COMPARISON TABLE" << std::endl;
    std::cout << "-------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(55) << "Implementation" 
              << std::right << std::setw(14) << "Time (ms)" 
              << std::right << std::setw(20) << "Speedup" << std::endl;
    std::cout << "-------------------------------------------------------------------------------" << std::endl;
    
    // Sort results by execution time (fastest first)
    std::vector<BenchmarkResult> sortedResults = benchmarkResults;
    std::sort(sortedResults.begin(), sortedResults.end(), 
              [](const BenchmarkResult& a, const BenchmarkResult& b) {
                  return a.averageDuration < b.averageDuration;
              });
    
    // Print all results with formatting
        for (const auto& result : sortedResults) {
        std::cout << std::left << std::setw(55) << result.methodName 
                  << std::right << std::fixed << std::setprecision(3) 
                  << std::setw(14) << result.averageDuration;
        
            if (!GLOBAL_BASELINE_NAME.empty() && result.methodName == GLOBAL_BASELINE_NAME) {
                std::cout << std::right << std::setw(20) << "baseline";
            } else {
                std::cout << std::right << std::setw(20) << (std::to_string(result.speedup) + "x faster");
            }
        std::cout << std::endl;
    }
    
    // Find and report the fastest implementation
    double bestSpeedup = 0.0;
    std::string fastestMethod;
    for (const auto& result : benchmarkResults) {
        if (result.speedup > bestSpeedup) {
            bestSpeedup = result.speedup;
            fastestMethod = result.methodName;
        }
    }
    
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Fastest implementation: " << fastestMethod << " (" 
              << std::fixed << std::setprecision(2) << bestSpeedup << "x faster than " << GLOBAL_BASELINE_NAME << ")" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
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
    const float absoluteEpsilon = 1e-6f;  // For small absolute differences
    const float relativeEpsilon = 1e-5f;  // For relative differences
    
    // Handle exact equality and very small numbers
    if (a == b || (std::fabs(a) < absoluteEpsilon && std::fabs(b) < absoluteEpsilon)) {
        return true;
    }
    
    // Check relative error for larger numbers
    return std::fabs(a - b) <= relativeEpsilon * std::max(std::fabs(a), std::fabs(b));
}

/**
 * Main program entry point.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return 0 on successful execution, 1 on error
 */
int main(int argc, char* argv[]) {
    try {
        log("Matrix multiplication benchmark program starting");
        
        // Process command line arguments for matrix dimensions
        // - positional numeric argument sets N (matrix dimension)
        // - --list prints available methods and exits
        // - --run=name1,name2,... or --run name1,name2,... selects methods to run
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
                    } catch (const std::exception& e) {
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
                    } catch (const std::exception& e) {
                        // ignore
                    }
                } else if (a.rfind("--run=", 0) == 0) {
                    std::string payload = a.substr(6);
                    std::stringstream ss(payload);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        if (!token.empty()) methodsToRun.insert(token);
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
                        if (!token.empty()) methodsToRun.insert(token);
                    }
                } else {
                    // If argument doesn't look like a flag, try to parse it as matrix size
                    bool looksLikeNumber = !a.empty();
                    for (char c : a) {
                        if (!std::isdigit(static_cast<unsigned char>(c))) { looksLikeNumber = false; break; }
                    }
                    if (looksLikeNumber) {
                        try {
                            int parsedN = std::stoi(a);
                            if (parsedN <= 0) {
                                std::cerr << "ERROR: Matrix dimension must be positive. Using default N = 512" << std::endl;
                            } else {
                                N = parsedN;
                                std::cout << "Using matrix dimension N = " << N << " from command line argument" << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "ERROR: Invalid matrix dimension: '" << a 
                                    << "'. Using default N = 512" << std::endl;
                            N = 512;
                        }
                    } else {
                        // Unrecognized token; ignore but log when verbose
                        log("Ignoring unrecognized command-line token: " + a);
                    }
                }
            }
        } else {
            // No arguments: print usage and available methods, then exit
            std::cout << "No command-line arguments detected.\n\n";
            std::cout << "Usage:\n";
            std::cout << "  [N]                : set matrix dimension (positive integer)\n";
            std::cout << "  --threads [T]      : set number of threads (positive integer)\n";
            std::cout << "  --list             : list available multiplication methods\n";
            std::cout << "  --run=name1,name2  : run only the named methods (comma-separated)\n\n";
            std::cout << "Available multiplication methods:\n";
            std::cout << "  Naive-ijkLoop\n";
            std::cout << "  Naive-ijkLoop-Parallel\n";
            std::cout << "  BlockTiled-CacheAware\n";
            std::cout << "  BlockTiled-CacheAware-Parallel\n";
            std::cout << "  SIMD-AVX2-Transposed\n";
            std::cout << "  RowColumn-Transposed\n";
            std::cout << "  Scalar-LoopUnrolled\n";
            std::cout << "  Parallel-SIMD-AVX2\n";
            std::cout << "  Parallel-Scalar-LoopUnrolled\n";
            std::cout << "  SIMD-AVX2-Direct\n";
            std::cout << "  Parallel-SIMD-Direct\n";
            std::cout << "  BlockLocal-StackTranspose\n\n";
            std::cout << "Example:\n";
            std::cout << "  ./bench-matmul 1024 --run=BlockTiled-CacheAware,SIMD-AVX2-Transposed\n\n";
            return 0;
        }
        
        // If user requested just the list, print it now and exit before any heavy work
        if (listOnly) {
            std::cout << "Available multiplication methods:\n";
            std::cout << "  Naive-ijkLoop\n";
            std::cout << "  Naive-ijkLoop-Parallel\n";
            std::cout << "  BlockTiled-CacheAware\n";
            std::cout << "  BlockTiled-CacheAware-Parallel\n";
            std::cout << "  SIMD-AVX2-Transposed\n";
            std::cout << "  RowColumn-Transposed\n";
            std::cout << "  Scalar-LoopUnrolled\n";
            std::cout << "  Parallel-SIMD-AVX2\n";
            std::cout << "  Parallel-Scalar-LoopUnrolled\n";
            std::cout << "  SIMD-AVX2-Direct\n";
            std::cout << "  Parallel-SIMD-Direct\n";
            std::cout << "  BlockLocal-StackTranspose\n";
            return 0;
        }

        log("Matrix dimension set to N = " + std::to_string(N));
        
        // Check if dimension is optimal (power of 2)
        bool isPowerOf2 = (N & (N - 1)) == 0;
        if (!isPowerOf2) {
            log("NOTE: N = " + std::to_string(N) + " is not a power of 2. Some optimizations may be less effective.");
        }
        
        // Initialize performance parameters
        log("Determining optimal block size based on CPU cache");
        determineOptimalBlockSize();
        
        // Adjust block size if necessary based on matrix dimension
        if (BLOCK_SIZE > N / 2) {
            BLOCK_SIZE = std::max(8, N / 4);
            log("Block size adjusted to " + std::to_string(BLOCK_SIZE) + " to better fit matrix dimensions");
        }
        
        if (!threadsSetByUser) {
            log("Determining optimal thread count based on CPU cores");
            determineOptimalThreadCount();
        } else {
            log("Thread count set to " + std::to_string(NUM_THREADS) + " by user");
            std::cout << "Using " << NUM_THREADS << " threads for parallel operations (set by user)" << std::endl;
        }
        
        std::cout << "Matrix Multiplication with Random Matrices (N=" << N << ", BLOCK_SIZE=" << BLOCK_SIZE << ")\n\n";
        
        // Initialize random number generator
        log("Setting up random number generation");
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(1.0f, 10.0f);
        
        // Allocate memory for matrices
        log("Allocating matrices");
        float** aMatrix = allocateMatrix();
        float** bMatrix = allocateMatrix();
        float** product = allocateMatrix();
        float** productNaiveParallel = allocateMatrix();
        float** productOptimized = allocateMatrix();
        float** productOptimizedParallel = allocateMatrix();
        float** productTransposed = allocateMatrix();
        float** productAVX2 = allocateMatrix();
        float** productThreaded = allocateMatrix();
        float** productOptimizedNoSIMD = allocateMatrix();
        float** productOptimizedNoSIMDThreaded = allocateMatrix();
        float** productAVX2NoTranspose = allocateMatrix();
        float** productThreadedAVX2NoTranspose = allocateMatrix();
        float** productLocalTranspose = allocateMatrix();
        
        // Initialize input matrices with random values
        log("Initializing matrices with random values");
        initializeRandomMatrixFast(aMatrix, gen, dist);
        initializeRandomMatrixFast(bMatrix, gen, dist);
        
        // Map available methods to function pointers and product buffers
        std::vector<std::pair<std::string, void(*)(float**, float**, float**)>> allMethods = {
            {"Naive-ijkLoop", multiplyMatrices},
            {"Naive-ijkLoop-Parallel", multiplyMatricesNaiveParallel},
            {"BlockTiled-CacheAware", multiplyMatricesOptimized},
            {"BlockTiled-CacheAware-Parallel", multiplyMatricesOptimizedParallel},
            {"SIMD-AVX2-Transposed", multiplyMatricesAVX2},
            {"RowColumn-Transposed", multiplyMatricesTransposed},
            {"Scalar-LoopUnrolled", multiplyMatricesOptimizedNoSIMD},
            {"Parallel-SIMD-AVX2", multiplyMatricesThreaded},
            {"Parallel-Scalar-LoopUnrolled", multiplyMatricesOptimizedNoSIMDThreaded},
            {"SIMD-AVX2-Direct", multiplyMatricesAVX2NoTranspose},
            {"Parallel-SIMD-Direct", multiplyMatricesThreadedAVX2NoTranspose},
            {"BlockLocal-StackTranspose", multiplyMatricesLocalTranspose}
        };
        
        // Map method names to product buffers (so benchmarks write to correct matrices)
        std::unordered_map<std::string, float**> productMap = {
            {"Naive-ijkLoop", product},
            {"Naive-ijkLoop-Parallel", productNaiveParallel},
            {"BlockTiled-CacheAware", productOptimized},
            {"BlockTiled-CacheAware-Parallel", productOptimizedParallel},
            {"SIMD-AVX2-Transposed", productAVX2},
            {"RowColumn-Transposed", productTransposed},
            {"Scalar-LoopUnrolled", productOptimizedNoSIMD},
            {"Parallel-SIMD-AVX2", productThreaded},
            {"Parallel-Scalar-LoopUnrolled", productOptimizedNoSIMDThreaded},
            {"SIMD-AVX2-Direct", productAVX2NoTranspose},
            {"Parallel-SIMD-Direct", productThreadedAVX2NoTranspose},
            {"BlockLocal-StackTranspose", productLocalTranspose}
        };
        
        // If user requested list, print available methods and exit
        if (listOnly) {
            std::cout << "Available multiplication methods:\n";
            for (const auto &m : allMethods) {
                std::cout << "  " << m.first << "\n";
            }
            // Clean up before exit
            deallocateMatrix(aMatrix);
            deallocateMatrix(bMatrix);
            deallocateMatrix(product);
            deallocateMatrix(productNaiveParallel);
            deallocateMatrix(productOptimized);
            deallocateMatrix(productOptimizedParallel);
            deallocateMatrix(productTransposed);
            deallocateMatrix(productAVX2);
            deallocateMatrix(productThreaded);
            deallocateMatrix(productOptimizedNoSIMD);
            deallocateMatrix(productOptimizedNoSIMDThreaded);
            deallocateMatrix(productAVX2NoTranspose);
            deallocateMatrix(productThreadedAVX2NoTranspose);
            deallocateMatrix(productLocalTranspose);
            checkMemoryLeaks();
            return 0;
        }
        
        // If the user specified methods, do not force-add a baseline.
        // The program will use the first executed method as the baseline if
        // `Naive-ijkLoop` is not present. This avoids surprising runs when
        // users explicitly request a subset of algorithms.
        
        // If user did not restrict methods, run all
        auto shouldRun = [&](const std::string &name) -> bool {
            return methodsToRun.empty() || methodsToRun.find(name) != methodsToRun.end();
        };

        // Determine baseline policy
        std::string baselineName = "";
        if (!baselineRequested.empty()) {
            // User explicitly requested a baseline; validate it exists
            bool found = false;
            for (const auto &m : allMethods) if (m.first == baselineRequested) { found = true; break; }
            if (!found) {
                std::cerr << "ERROR: Requested baseline '" << baselineRequested << "' is not a known method.\n";
                std::cerr << "Available methods:\n";
                for (const auto &m : allMethods) std::cerr << "  " << m.first << "\n";
                return 1;
            }
            // Ensure baseline is included in the run set
            if (methodsToRun.empty() || methodsToRun.find(baselineRequested) == methodsToRun.end()) {
                methodsToRun.insert(baselineRequested);
            }
            baselineName = baselineRequested;
        } else if (shouldRun("Naive-ijkLoop")) {
            // Prefer naive baseline when it will be run
            baselineName = "Naive-ijkLoop";
        } else {
            baselineName = ""; // will pick first executed method as baseline
        }

        // Build ordered list of methods to execute: baseline first (if any), then the rest
        std::vector<std::pair<std::string, void(*)(float**, float**, float**)>> execOrder;
        if (!baselineName.empty()) {
            // push baseline first
            for (const auto &m : allMethods) {
                if (m.first == baselineName) { execOrder.push_back(m); break; }
            }
        }
        for (const auto &m : allMethods) {
            if (!shouldRun(m.first)) continue;
            if (!baselineName.empty() && m.first == baselineName) continue;
            execOrder.push_back(m);
        }

        // If baselineName determined, set GLOBAL_BASELINE_NAME so reporting uses it
        if (!baselineName.empty()) GLOBAL_BASELINE_NAME = baselineName;

        // Execute benchmarks for selected implementations in the chosen order
        log("Beginning benchmark sequence");
        for (const auto &entry : execOrder) {
            const std::string &name = entry.first;
            auto func = entry.second;
            // find the corresponding product buffer
            float** outBuf = productMap[name];
            benchmarkMultiplication(aMatrix, bMatrix, outBuf, func, name);
        }
        
        // Verify correctness of all implementations that were run
        log("Verifying correctness of results from selected multiplication methods");
        // Identify baseline pointer (must exist after benchmark sequence).
        // The baseline name is chosen by the benchmark runner and stored in GLOBAL_BASELINE_NAME.
        float** baselineBuf = productMap[GLOBAL_BASELINE_NAME];
        bool isEqual = true;
        // Build list of methods that were run (excluding baseline)
        std::vector<std::string> runMethods;
        for (const auto &entry : allMethods) {
            if (shouldRun(entry.first) && entry.first != GLOBAL_BASELINE_NAME) {
                runMethods.push_back(entry.first);
            }
        }
        
        for (int i = 0; i < N; i++) {
            if (i % 1000 == 0) {
                log("Verification progress: checking row " + std::to_string(i) + " of " + std::to_string(N));
            }
            for (int j = 0; j < N; j++) {
                for (const auto &mname : runMethods) {
                    float** cmpBuf = productMap[mname];
                    if (!almostEqual(baselineBuf[i][j], cmpBuf[i][j])) {
                        isEqual = false;
                        log("ALERT: Discrepancy found at position [" + std::to_string(i) + "][" + std::to_string(j) + "] between baseline and " + mname);
                        std::cout << "Discrepancy at [" << i << "][" << j << "]: baseline=" << baselineBuf[i][j]
                                  << "  " << mname << "=" << cmpBuf[i][j] << std::endl;
                        throw std::runtime_error("Result mismatch detected at position [" + std::to_string(i) + "][" + std::to_string(j) + "]");
                    }
                }
            }
        }
        
        // This code will only be reached if there were no mismatches
        log("Verification successful: All implementations produced identical results");
        
        // Generate performance report
        log("Generating performance summary");
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "PERFORMANCE SUMMARY" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Matrix size: " << N << " x " << N << std::endl;
        std::cout << "Block size: " << BLOCK_SIZE << std::endl;
        std::cout << "CPU threads: " << NUM_THREADS << " (of " << std::thread::hardware_concurrency() << " available)" << std::endl;
        
        // Calculate and report memory usage
        double memoryUsageKB = (matrixAllocations * sizeof(float) * N * N) / 1024.0;
        double memoryUsageMB = memoryUsageKB / 1024.0;
        std::cout << "Total memory usage: " << std::fixed << std::setprecision(2);
        if (memoryUsageMB >= 1.0) {
            std::cout << memoryUsageMB << " MB" << std::endl;
        } else {
            std::cout << memoryUsageKB << " KB" << std::endl;
        }
        
        std::cout << "----------------------------------------" << std::endl;
        
        // Show detailed performance comparison
        displayPerformanceComparisonTable();
        
        // Clean up allocated memory
        log("Deallocating matrices");
        deallocateMatrix(aMatrix);
        deallocateMatrix(bMatrix);
        deallocateMatrix(product);
        deallocateMatrix(productNaiveParallel);
        deallocateMatrix(productOptimized);
        deallocateMatrix(productTransposed);
        deallocateMatrix(productAVX2);
        deallocateMatrix(productThreaded);
        deallocateMatrix(productOptimizedNoSIMD);
        deallocateMatrix(productOptimizedNoSIMDThreaded);
        deallocateMatrix(productAVX2NoTranspose);
        deallocateMatrix(productThreadedAVX2NoTranspose);
        deallocateMatrix(productLocalTranspose);
        deallocateMatrix(productOptimizedParallel);
        
        // Verify proper memory management
        log("Checking for memory leaks");
        checkMemoryLeaks();
        
        log("Program completed successfully");
        return 0;
    } 
    catch (const std::exception& e) {
        // Handle known exceptions
        log("EXCEPTION: " + std::string(e.what()));
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    } 
    catch (...) {
        // Handle unknown exceptions
        log("EXCEPTION: Unknown exception occurred");
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
}
