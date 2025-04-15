#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <immintrin.h> // For AVX2 intrinsics
#include <thread>
#include <vector>
#include <cstring> // For memset

// Add this after the includes for logging control
bool VERBOSE_LOGGING = true;  // Global flag to enable/disable detailed logging

// Logger helper function
void log(const std::string& message) {
    if (VERBOSE_LOGGING) {
        std::cout << "[LOG] " << message << std::endl;
    }
}

// Change from define to global variable
int N = 512;  // Default value, will be overridden by command-line argument if provided
int BLOCK_SIZE = 32;  // Default block size, will adjust if possible

// Global variable for optimal thread count
unsigned int NUM_THREADS = 1;  // Default value, will be adjusted

// Function to determine optimal number of threads
void determineOptimalThreadCount() {
    // Get hardware concurrency
    unsigned int hwThreads = std::thread::hardware_concurrency();
    
    // Ensure at least 2 threads and no more than reasonable for the problem size
    NUM_THREADS = std::max(1u, std::min(hwThreads, 48u));
    
    log("Hardware detection: " + std::to_string(hwThreads) + " logical CPU cores available");
    if (NUM_THREADS != hwThreads) {
        log("Thread count adjusted to " + std::to_string(NUM_THREADS) + " for optimal performance");
    }
    
    std::cout << "Hardware supports " << hwThreads << " concurrent threads" << std::endl;
    std::cout << "Using " << NUM_THREADS << " threads for parallel operations" << std::endl;
}

// Function to determine optimal block size based on CPU cache size
void determineOptimalBlockSize() {
    int cacheSizeKB = 0;
    
    // Try to read L1 cache size from Linux's sysfs (typical on most Linux systems)
    std::ifstream cacheFile("/sys/devices/system/cpu/cpu0/cache/index1/size");
    if (cacheFile.is_open()) {
        std::string sizeStr;
        cacheFile >> sizeStr;
        cacheSizeKB = std::stoi(sizeStr);
        cacheFile.close();
    } else {
        // Try reading from CPU-specific files or use a default
        // For example, try L2 cache which is likely larger
        std::ifstream cacheFileL2("/sys/devices/system/cpu/cpu0/cache/index2/size");
        if (cacheFileL2.is_open()) {
            std::string sizeStr;
            cacheFileL2 >> sizeStr;
            cacheSizeKB = std::stoi(sizeStr);
            cacheFileL2.close();
        } else {
            // If we can't detect, use a reasonable default for modern CPUs
            cacheSizeKB = 256;  // Assume 256KB L2 cache as a safe default
        }
    }
    
    // Convert KB to bytes
    int cacheSizeBytes = cacheSizeKB * 1024;
    
    // Calculate block size to fit 3 blocks (A, B, result) in the cache
    // Each block is block_size^2 elements of size sizeof(int) bytes
    int maxMatrixSizeBytes = cacheSizeBytes / 3;
    int maxBlockElements = maxMatrixSizeBytes / sizeof(int);
    int optimalBlockSize = static_cast<int>(std::sqrt(maxBlockElements));
    
    // Round down to a power of 2 for better memory alignment
    int powerOf2 = 1;
    while (powerOf2 * 2 <= optimalBlockSize) {
        powerOf2 *= 2;
    }
    
    // Set the global block size
    BLOCK_SIZE = powerOf2;
    
    std::cout << "Cache size detected: " << cacheSizeKB << "KB" << std::endl;
    std::cout << "Optimal block size: " << BLOCK_SIZE << std::endl;
}

// Memory tracking variables
int matrixAllocations = 0;
int matrixDeallocations = 0;

// Function to allocate a matrix on the heap using contiguous memory
float** allocateMatrix() {
    try {
        log("Allocating matrix of size " + std::to_string(N) + "x" + std::to_string(N));
        
        // Allocate contiguous memory for the entire matrix data
        float* data = new float[N * N];
        
        // Allocate array of pointers to rows
        float** matrix = new float*[N];
        
        // Set up row pointers to the appropriate positions in the data block
        for (int i = 0; i < N; i++) {
            matrix[i] = &data[i * N];
        }
        
        // Track this allocation
        matrixAllocations++;
        log("Matrix allocated successfully (total: " + std::to_string(matrixAllocations) + ")");
        
        return matrix;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to deallocate a matrix from heap (for contiguous storage)
void deallocateMatrix(float** matrix) {
    if (matrix == nullptr) {
        log("Warning: Attempted to deallocate nullptr matrix");
        return; // Guard against null pointer
    }
    
    // Delete the matrix data (first row contains pointer to the contiguous block)
    delete[] matrix[0];
    
    // Delete the array of row pointers
    delete[] matrix;
    
    // Track this deallocation
    matrixDeallocations++;
    log("Matrix deallocated successfully (total: " + std::to_string(matrixDeallocations) + ")");
}

// Function to check for memory leaks
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

// Function to initialize a matrix with random values
void initializeRandomMatrix(float** matrix, std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = dist(gen);
        }
    }
}

// Optimized function to initialize a matrix with random values
void initializeRandomMatrixFast(float** matrix, std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    // Access in contiguous memory order for better cache usage
    // Since we're using a 1D array under the hood, we can access it directly
    float* data = matrix[0];  // Points to the start of the contiguous block
    
    // Single loop is more cache-friendly than nested loops
    for (int i = 0; i < N * N; i++) {
        data[i] = dist(gen);
    }
}

// Function to print a matrix
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

// Function to perform standard matrix multiplication
void multiplyMatrices(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros - use memset for efficiency
    memset(result[0], 0, N * N * sizeof(float));
    
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            for (int inner = 0; inner != N; inner++) {
                result[row][col] += matrixA[row][inner] * matrixB[inner][col];
            }
        }
    }
}

// Function to perform optimized matrix multiplication using cache blocking/tiling
void multiplyMatricesOptimized(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));
    
    // Process all blocks
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // Process all corresponding blocks for this result block
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Process current block
                for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                    for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                        // Load matrixA[i][k] into register once per inner loop
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

// Function to perform matrix multiplication with transposition and tiling
void multiplyMatricesTransposed(float** matrixA, float** matrixB, float** result) {
    // First create a transposed copy of matrix B for better cache locality
    float** transposedB = allocateMatrix();
    
    // Transpose B in a cache-friendly way (using tiling)
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                    transposedB[j][i] = matrixB[i][j];
                }
            }
        }
    }
    
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));
    
    // Perform tiled multiplication with the transposed B matrix
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Process current block
                for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        float sum = 0;
                        // Now we can access both matrices in row-major order
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            sum += matrixA[i][k] * transposedB[j][k];
                        }
                        result[i][j] += sum;
                    }
                }
            }
        }
    }
    
    // Clean up the temporary transposed matrix
    deallocateMatrix(transposedB);
}

// Function to perform SIMD-optimized matrix multiplication using AVX2 instructions
void multiplyMatricesAVX2(float** matrixA, float** matrixB, float** result) {
    // First create a transposed copy of matrix B for better cache locality
    float** transposedB = allocateMatrix();
    
    // Transpose B in a cache-friendly way (using tiling)
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                    transposedB[j][i] = matrixB[i][j];
                }
            }
        }
    }
    
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));

    // Process in blocks for better cache locality
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // Process current block
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                // Process 8 elements at once with AVX2
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize accumulator registers to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate results across all k values
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcasting element from A
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from transposed B
                            __m256 b_vals = _mm256_setr_ps(
                                transposedB[j][k], transposedB[j+1][k], 
                                transposedB[j+2][k], transposedB[j+3][k],
                                transposedB[j+4][k], transposedB[j+5][k], 
                                transposedB[j+6][k], transposedB[j+7][k]
                            );
                            
                            // Use FMA to accumulate results in registers
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store accumulated results back to memory (only once per block)
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining elements (if N is not divisible by 8)
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
    
    // Clean up the temporary transposed matrix
    deallocateMatrix(transposedB);
}

// Function to perform SIMD-optimized matrix multiplication using AVX2 instructions
// without transposing matrix B (direct access)
void multiplyMatricesAVX2NoTranspose(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));

    // Process in blocks for better cache locality
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // Process current block
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                // Process 8 elements at once with AVX2
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize accumulator registers to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate results across all k values
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcasting element from A
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from B - direct access without transposition
                            // This requires 8 separate loads, which is less cache efficient
                            __m256 b_vals = _mm256_setr_ps(
                                matrixB[k][j], matrixB[k][j+1], 
                                matrixB[k][j+2], matrixB[k][j+3],
                                matrixB[k][j+4], matrixB[k][j+5], 
                                matrixB[k][j+6], matrixB[k][j+6]
                            );
                            
                            // Use FMA to accumulate results in registers
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store accumulated results back to memory (only once per block)
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining elements (if N is not divisible by 8)
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

// Function to perform multithreaded matrix multiplication with SIMD and transposition
void multiplyMatricesThreaded(float** matrixA, float** matrixB, float** result) {
    // First create a transposed copy of matrix B for better cache locality
    float** transposedB = allocateMatrix();
    
    // Transpose B in a cache-friendly way (using tiling) - this is still sequential
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                    transposedB[j][i] = matrixB[i][j];
                }
            }
        }
    }
    
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));

    // Function for each thread to process a subset of rows
    auto threadFunction = [&](int startRow, int endRow, int threadId) {
        for (int i = startRow; i < endRow; i++) {
            // Process this row in blocks for better cache locality
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                // Process 8 elements at once with AVX2
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize accumulator registers to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate results across all k values
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcasting element from A
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from transposed B
                            __m256 b_vals = _mm256_setr_ps(
                                transposedB[j][k], transposedB[j+1][k], 
                                transposedB[j+2][k], transposedB[j+3][k],
                                transposedB[j+4][k], transposedB[j+5][k], 
                                transposedB[j+6][k], transposedB[j+7][k]
                            );
                            
                            // Use FMA to accumulate results in registers
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store accumulated results back to memory (only once per block)
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining elements (if N is not divisible by 8)
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
    
    // Create and launch threads
    std::vector<std::thread> threads;
    int rowsPerThread = N / NUM_THREADS;
    
    for (unsigned int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? N : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(threadFunction, startRow, endRow, t));
    }
    
    // Wait for all threads to finish
    for (unsigned int t = 0; t < threads.size(); t++) {
        threads[t].join();
    }
    
    // Clean up the temporary transposed matrix
    deallocateMatrix(transposedB);
}

// Function to perform highly optimized matrix multiplication without SIMD intrinsics
void multiplyMatricesOptimizedNoSIMD(float** matrixA, float** matrixB, float** result) {
    // First create a transposed copy of matrix B for better cache locality
    float** transposedB = allocateMatrix();
    
    // Transpose B in a cache-friendly way (using tiling)
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                    transposedB[j][i] = matrixB[i][j];
                }
            }
        }
    }
    
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));
    
    // Cache-aware multiplication with loop interchange for better performance
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Process current block with loop interchange
                for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        // Use a register to accumulate results
                        float sum = result[i][j];
                        
                        // Unroll the inner loop by 4 for better instruction-level parallelism
                        int k = bk;
                        for (; k < std::min(bk + BLOCK_SIZE, N) - 3; k += 4) {
                            // Manual loop unrolling for better instruction pipelining
                            sum += matrixA[i][k] * transposedB[j][k];
                            sum += matrixA[i][k+1] * transposedB[j][k+1];
                            sum += matrixA[i][k+2] * transposedB[j][k+2];
                            sum += matrixA[i][k+3] * transposedB[j][k+3];
                        }
                        
                        // Handle remaining elements
                        for (; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            sum += matrixA[i][k] * transposedB[j][k];
                        }
                        
                        // Store the result
                        result[i][j] = sum;
                    }
                }
            }
        }
    }
    
    // Clean up the temporary transposed matrix
    deallocateMatrix(transposedB);
}

// Function to perform multithreaded optimized matrix multiplication without SIMD intrinsics
void multiplyMatricesOptimizedNoSIMDThreaded(float** matrixA, float** matrixB, float** result) {
    
    // First create a transposed copy of matrix B for better cache locality
    float** transposedB = allocateMatrix();
    
    // Transpose B in a cache-friendly way (using tiling) - this is still sequential
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                    transposedB[j][i] = matrixB[i][j];
                }
            }
        }
    }
    
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));

    // Function for each thread to process a subset of rows
    auto threadFunction = [&](int startRow, int endRow, int threadId) {
        
        // Cache-aware multiplication with loop interchange for better performance
        for (int i = startRow; i < endRow; i++) {
            // Process this row in blocks for better cache locality
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                    // Process current block with loop interchange
                    for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                        // Use a register to accumulate results
                        float sum = result[i][j];
                        
                        // Unroll the inner loop by 4 for better instruction-level parallelism
                        int k = bk;
                        for (; k < std::min(bk + BLOCK_SIZE, N) - 3; k += 4) {
                            // Manual loop unrolling for better instruction pipelining
                            sum += matrixA[i][k] * transposedB[j][k];
                            sum += matrixA[i][k+1] * transposedB[j][k+1];
                            sum += matrixA[i][k+2] * transposedB[j][k+2];
                            sum += matrixA[i][k+3] * transposedB[j][k+3];
                        }
                        
                        // Handle remaining elements
                        for (; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            sum += matrixA[i][k] * transposedB[j][k];
                        }
                        
                        // Store the result
                        result[i][j] = sum;
                    }
                }
            }
        }
    };
    
    // Create and launch threads
    std::vector<std::thread> threads;
    int rowsPerThread = N / NUM_THREADS;
    
    for (unsigned int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? N : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(threadFunction, startRow, endRow, t));
    }
    
    // Wait for all threads to finish
    for (unsigned int t = 0; t < threads.size(); t++) {
        threads[t].join();
    }
    
    // Clean up the temporary transposed matrix
    deallocateMatrix(transposedB);
}

// Function to perform multithreaded matrix multiplication with AVX2 instructions
// without transposing matrix B (direct access)
void multiplyMatricesThreadedAVX2NoTranspose(float** matrixA, float** matrixB, float** result) {
    // Initialize result matrix to zeros using memset
    memset(result[0], 0, N * N * sizeof(float));

    // Function for each thread to process a subset of rows
    auto threadFunction = [&](int startRow, int endRow, int threadId) {
        for (int i = startRow; i < endRow; i++) {
            // Process this row in blocks for better cache locality
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                // Process 8 elements at once with AVX2
                for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
                    // Initialize accumulator registers to zero
                    __m256 c_vals = _mm256_setzero_ps();
                    
                    // Accumulate results across all k values
                    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
                            // Broadcasting element from A
                            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
                            
                            // Load 8 elements from B - direct access without transposition
                            __m256 b_vals = _mm256_setr_ps(
                                matrixB[k][j], matrixB[k][j+1], 
                                matrixB[k][j+2], matrixB[k][j+3],
                                matrixB[k][j+4], matrixB[k][j+5], 
                                matrixB[k][j+6], matrixB[k][j+7]
                            );
                            
                            // Use FMA to accumulate results in registers
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                        }
                    }
                    
                    // Store accumulated results back to memory (only once per block)
                    _mm256_storeu_ps(&result[i][j], c_vals);
                }
                
                // Handle remaining elements (if N is not divisible by 8)
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
    
    // Create and launch threads
    std::vector<std::thread> threads;
    int rowsPerThread = N / NUM_THREADS;
    
    for (unsigned int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? N : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(threadFunction, startRow, endRow, t));
    }
    
    // Wait for all threads to finish
    for (unsigned int t = 0; t < threads.size(); t++) {
        threads[t].join();
    }
}

// Add this structure to store benchmark results
struct BenchmarkResult {
    std::string methodName;
    double averageDuration;  // in milliseconds
    double speedup;          // relative to baseline
};

// Add a global vector to store all benchmark results
std::vector<BenchmarkResult> benchmarkResults;

// Update the benchmarking function to store results
void benchmarkMultiplication(float** matrixA, float** matrixB, float** result, 
                            void (*multiplyFunc)(float**, float**, float**),
                            const std::string& methodName) {
    log("Beginning benchmark of " + methodName);
    
    // Run multiple iterations for more stable measurements
    const int iterations = 3;  // Increased from 1 to 3 for better reliability
    double totalDuration = 0.0;
    
    for (int iter = 0; iter < iterations; iter++) {
        log("  Running iteration " + std::to_string(iter+1) + " of " + std::to_string(iterations));
        
        // Use high resolution clock for timing
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run the multiplication function
        multiplyFunc(matrixA, matrixB, result);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double duration_ms = duration_us.count() / 1000.0;
        
        log("  Iteration " + std::to_string(iter+1) + " completed in " + 
            std::to_string(duration_ms) + " ms");
        
        // Convert to milliseconds and add to total
        totalDuration += duration_ms;
    }
    
    // Calculate average duration in milliseconds
    double avgDuration = totalDuration / iterations;
    log("Average execution time for " + methodName + ": " + std::to_string(avgDuration) + " ms");
    
    // Calculate speedup compared to baseline
    static double baselineDuration = -1;
    double speedup = 1.0;
    
    if (methodName == "Standard multiplication") {
        baselineDuration = avgDuration;
    } else if (baselineDuration > 0) {
        speedup = baselineDuration / avgDuration;
    }
    
    // Store results for final summary table
    BenchmarkResult benchmarkResult;
    benchmarkResult.methodName = methodName;
    benchmarkResult.averageDuration = avgDuration;
    benchmarkResult.speedup = speedup;
    benchmarkResults.push_back(benchmarkResult);
    
    log("Benchmark of " + methodName + " completed");
}

// Add this function to display the final performance comparison table
void displayPerformanceComparisonTable() {
    log("Generating performance comparison table");
    
    // Print table header with clear formatting
    std::cout << "\n-----------------------------------------------------------------------------" << std::endl;
    std::cout << "PERFORMANCE COMPARISON TABLE" << std::endl;
    std::cout << "-------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(55) << "Implementation" 
              << std::right << std::setw(14) << "Time (ms)" 
              << std::right << std::setw(20) << "Speedup" << std::endl;
    std::cout << "-------------------------------------------------------------------------------" << std::endl;
    
    // Print each result in the table, sorted by speed (fastest first)
    std::vector<BenchmarkResult> sortedResults = benchmarkResults;
    std::sort(sortedResults.begin(), sortedResults.end(), 
              [](const BenchmarkResult& a, const BenchmarkResult& b) {
                  return a.averageDuration < b.averageDuration;
              });
    
    for (const auto& result : sortedResults) {
        std::cout << std::left << std::setw(55) << result.methodName 
                  << std::right << std::fixed << std::setprecision(3) 
                  << std::setw(14) << result.averageDuration;
        
        if (result.methodName == "Standard multiplication") {
            std::cout << std::right << std::setw(20) << "baseline";
        } else {
            std::cout << std::right << std::setw(20) << (std::to_string(result.speedup) + "x faster");
        }
        std::cout << std::endl;
    }
    
    // Calculate best speedup
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
              << std::fixed << std::setprecision(2) << bestSpeedup << "x faster than baseline)" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
}

// Add this function above main()
bool almostEqual(float a, float b) {
    const float absoluteEpsilon = 1e-6f;
    const float relativeEpsilon = 1e-5f;
    
    // Handle exact equality and very small numbers
    if (a == b || (std::fabs(a) < absoluteEpsilon && std::fabs(b) < absoluteEpsilon)) {
        return true;
    }
    
    // Check relative error
    return std::fabs(a - b) <= relativeEpsilon * std::max(std::fabs(a), std::fabs(b));
}

// Update main function to accept arguments
int main(int argc, char* argv[]) {
    try {
        log("Matrix multiplication benchmark program starting");
        
        // Parse command line arguments for matrix dimensions
        if (argc > 1) {
            try {
                N = std::stoi(argv[1]);
                if (N <= 0) {
                    std::cerr << "ERROR: Matrix dimension must be positive. Using default N = 512" << std::endl;
                    N = 512;
                } else {
                    std::cout << "Using matrix dimension N = " << N << " from command line argument" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Invalid matrix dimension: '" << argv[1] 
                          << "'. Using default N = 512" << std::endl;
                N = 512;
            }
        } else {
            std::cout << "No dimension specified. Using default N = 512" << std::endl;
        }
        
        log("Matrix dimension set to N = " + std::to_string(N));
        
        // Check if N is a power of 2 (optimal for many algorithms)
        bool isPowerOf2 = (N & (N - 1)) == 0;
        if (!isPowerOf2) {
            log("NOTE: N = " + std::to_string(N) + " is not a power of 2. Some optimizations may be less effective.");
        }
        
        // Determine the optimal block size based on CPU cache
        log("Determining optimal block size based on CPU cache");
        determineOptimalBlockSize();
        
        // Ensure BLOCK_SIZE is reasonable for the matrix size
        if (BLOCK_SIZE > N / 2) {
            BLOCK_SIZE = std::max(8, N / 4);
            log("Block size adjusted to " + std::to_string(BLOCK_SIZE) + " to better fit matrix dimensions");
        }
        
        // Determine optimal thread count
        log("Determining optimal thread count based on CPU cores");
        determineOptimalThreadCount();
        
        std::cout << "Matrix Multiplication with Random Matrices (N=" << N << ", BLOCK_SIZE=" << BLOCK_SIZE << ")\n\n";
        
        // Set up random number generation
        log("Setting up random number generation");
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(1.0f, 10.0f); // Random floats from 1.0 to 10.0
        
        // Allocate matrices on the heap
        log("Allocating matrices");
        float** aMatrix = allocateMatrix();
        float** bMatrix = allocateMatrix();
        float** product = allocateMatrix();
        float** productOptimized = allocateMatrix();
        float** productTransposed = allocateMatrix();
        float** productAVX2 = allocateMatrix();
        float** productThreaded = allocateMatrix();
        float** productOptimizedNoSIMD = allocateMatrix();
        float** productOptimizedNoSIMDThreaded = allocateMatrix();
        float** productAVX2NoTranspose = allocateMatrix();
        float** productThreadedAVX2NoTranspose = allocateMatrix();
        
        // Initialize matrices with random values
        log("Initializing matrices with random values");
        initializeRandomMatrixFast(aMatrix, gen, dist);
        initializeRandomMatrixFast(bMatrix, gen, dist);
        
        log("Beginning benchmark sequence");
        
        // Perform standard matrix multiplication with timing
        benchmarkMultiplication(aMatrix, bMatrix, product, multiplyMatrices, "Standard multiplication");
        
        // Perform optimized matrix multiplication with timing
        benchmarkMultiplication(aMatrix, bMatrix, productOptimized, multiplyMatricesOptimized, "Optimized multiplication");
        
        // Perform AVX2 matrix multiplication with timing
        benchmarkMultiplication(aMatrix, bMatrix, productAVX2, multiplyMatricesAVX2, "AVX2 multiplication");
        
        // Perform transposed tiled matrix multiplication with timing
        benchmarkMultiplication(aMatrix, bMatrix, productTransposed, multiplyMatricesTransposed, "Transposed + tiled multiplication");
        
        // Perform optimized matrix multiplication without SIMD with timing
        benchmarkMultiplication(aMatrix, bMatrix, productOptimizedNoSIMD, multiplyMatricesOptimizedNoSIMD, "Optimized multiplication without SIMD");

        // Perform multithreaded matrix multiplication with timing
        benchmarkMultiplication(aMatrix, bMatrix, productThreaded, multiplyMatricesThreaded, "Multithreaded multiplication");
        
        // Perform multithreaded optimized matrix multiplication without SIMD with timing
        benchmarkMultiplication(aMatrix, bMatrix, productOptimizedNoSIMDThreaded, multiplyMatricesOptimizedNoSIMDThreaded, "Multithreaded optimized multiplication without SIMD");
        
        // Perform AVX2 matrix multiplication without transposition with timing
        benchmarkMultiplication(aMatrix, bMatrix, productAVX2NoTranspose, multiplyMatricesAVX2NoTranspose, "AVX2 multiplication without transposition");
        
        // Perform multithreaded AVX2 matrix multiplication without transposition with timing
        benchmarkMultiplication(aMatrix, bMatrix, productThreadedAVX2NoTranspose, multiplyMatricesThreadedAVX2NoTranspose, "Multithreaded AVX2 multiplication without transposition");
        
        // Verify correctness
        log("Verifying correctness of results from all multiplication methods");
        bool isEqual = true;
        for (int i = 0; i < N && isEqual; i++) {
            if (i % 1000 == 0) {
                log("Verification progress: checking row " + std::to_string(i) + " of " + std::to_string(N));
            }
            for (int j = 0; j < N && isEqual; j++) {
                if (!almostEqual(product[i][j], productOptimized[i][j]) || 
                    !almostEqual(product[i][j], productTransposed[i][j]) || 
                    !almostEqual(product[i][j], productAVX2[i][j]) || 
                    !almostEqual(product[i][j], productThreaded[i][j]) || 
                    !almostEqual(product[i][j], productOptimizedNoSIMD[i][j]) || 
                    !almostEqual(product[i][j], productOptimizedNoSIMDThreaded[i][j]) ||
                    !almostEqual(product[i][j], productAVX2NoTranspose[i][j]) ||
                    !almostEqual(product[i][j], productThreadedAVX2NoTranspose[i][j])) {
                    isEqual = false;
                    log("ALERT: Discrepancy found at position [" + std::to_string(i) + "][" + std::to_string(j) + "]");
                    std::cout << "Discrepancy found at position [" << i << "][" << j << "]:\n" 
                              << "  Standard: " << product[i][j] << "\n"
                              << "  Tiled: " << productOptimized[i][j] << "\n"
                              << "  Transposed: " << productTransposed[i][j] << "\n"
                              << "  AVX2: " << productAVX2[i][j] << "\n"
                              << "  Multithreaded: " << productThreaded[i][j] << "\n"
                              << "  Optimized No SIMD: " << productOptimizedNoSIMD[i][j] << "\n"
                              << "  Multithreaded Optimized No SIMD: " << productOptimizedNoSIMDThreaded[i][j] << "\n"
                              << "  AVX2 No Transpose: " << productAVX2NoTranspose[i][j] << "\n"
                              << "  Multithreaded AVX2 No Transpose: " << productThreadedAVX2NoTranspose[i][j] << std::endl;
                }
            }
        }
        
        if (isEqual) {
            log("Verification successful: All implementations produced identical results");
        } else {
            log("ALERT: Verification failed - implementations produced different results");
        }
        
        // Display hardware info and summary
        log("Generating performance summary");
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "PERFORMANCE SUMMARY" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Matrix size: " << N << " x " << N << std::endl;
        std::cout << "Block size: " << BLOCK_SIZE << std::endl;
        std::cout << "CPU threads: " << NUM_THREADS << " (of " << std::thread::hardware_concurrency() << " available)" << std::endl;
        
        // Print memory usage
        double memoryUsageKB = (matrixAllocations * sizeof(float) * N * N) / 1024.0;
        double memoryUsageMB = memoryUsageKB / 1024.0;
        std::cout << "Total memory usage: " << std::fixed << std::setprecision(2);
        if (memoryUsageMB >= 1.0) {
            std::cout << memoryUsageMB << " MB" << std::endl;
        } else {
            std::cout << memoryUsageKB << " KB" << std::endl;
        }
        
        std::cout << "----------------------------------------" << std::endl;
        
        // Display the final performance comparison table
        displayPerformanceComparisonTable();
        
        // Deallocate matrices to prevent memory leaks
        log("Deallocating matrices");
        deallocateMatrix(aMatrix);
        deallocateMatrix(bMatrix);
        deallocateMatrix(product);
        deallocateMatrix(productOptimized);
        deallocateMatrix(productTransposed);
        deallocateMatrix(productAVX2);
        deallocateMatrix(productThreaded);
        deallocateMatrix(productOptimizedNoSIMD);
        deallocateMatrix(productOptimizedNoSIMDThreaded);
        deallocateMatrix(productAVX2NoTranspose);
        deallocateMatrix(productThreadedAVX2NoTranspose);
        
        // Check for memory leaks
        log("Checking for memory leaks");
        checkMemoryLeaks();
        
        log("Program completed successfully");
        return 0;
    } catch (const std::exception& e) {
        log("EXCEPTION: " + std::string(e.what()));
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        log("EXCEPTION: Unknown exception occurred");
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
}
