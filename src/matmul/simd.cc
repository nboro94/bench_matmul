#include "matmul.h"
#include "pthread_matmul.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h> // Intel SIMD intrinsics for AVX2 operations
#include <thread>
#include <vector>

void multiplyMatricesAVX2(float **matrixA, float **matrixB, float **result) {
  // Create transposed copy of B for better memory access patterns
  float **transposedB = allocateMatrix();
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
              __m256 b_vals =
                  _mm256_setr_ps(transposedB[j][k], transposedB[j + 1][k],
                                 transposedB[j + 2][k], transposedB[j + 3][k],
                                 transposedB[j + 4][k], transposedB[j + 5][k],
                                 transposedB[j + 6][k], transposedB[j + 7][k]);

              // Fused multiply-add: c_vals += a_val * b_vals
              c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
            }
          }

          // Store computed values back to result matrix
          _mm256_storeu_ps(&result[i][j], c_vals);
        }

        // Handle remaining columns (when N not divisible by 8)
        for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8);
             j < std::min(bj + BLOCK_SIZE, N); j++) {
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

void multiplyMatricesAVX2NoTranspose(float **matrixA, float **matrixB,
                                     float **result) {
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
                  matrixB[k][j], matrixB[k][j + 1], matrixB[k][j + 2],
                  matrixB[k][j + 3], matrixB[k][j + 4], matrixB[k][j + 5],
                  matrixB[k][j + 6], matrixB[k][j + 7]);

              // Fused multiply-add: c_vals += a_val * b_vals
              c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
            }
          }

          // Store computed values back to result matrix
          _mm256_storeu_ps(&result[i][j], c_vals);
        }

        // Handle remaining columns (when N not divisible by 8)
        for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8);
             j < std::min(bj + BLOCK_SIZE, N); j++) {
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

void multiplyMatricesThreaded(float **matrixA, float **matrixB,
                              float **result) {
  // Create transposed copy of B for better memory access patterns
  // Use parallel transposition for better performance
  float **transposedB = allocateMatrix();
  transposeMatrixParallel(matrixB, transposedB);

  // Initialize result matrix to zeros
  memset(result[0], 0, N * N * sizeof(float));

  // Define work function for each thread
  auto threadFunction = [&](int startRow, int endRow) {
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
              __m256 b_vals =
                  _mm256_setr_ps(transposedB[j][k], transposedB[j + 1][k],
                                 transposedB[j + 2][k], transposedB[j + 3][k],
                                 transposedB[j + 4][k], transposedB[j + 5][k],
                                 transposedB[j + 6][k], transposedB[j + 7][k]);

              // Fused multiply-add: c_vals += a_val * b_vals
              c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
            }
          }

          // Store computed values back to result matrix
          _mm256_storeu_ps(&result[i][j], c_vals);
        }

        // Handle remaining columns (when N not divisible by 8)
        for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8);
             j < std::min(bj + BLOCK_SIZE, N); j++) {
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
    threads.push_back(std::thread(threadFunction, startRow, endRow));
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Release temporary transposed matrix
  deallocateMatrix(transposedB);
}

void multiplyMatricesThreadedAVX2NoTranspose(float **matrixA, float **matrixB,
                                             float **result) {
  // Initialize result matrix to zeros
  memset(result[0], 0, N * N * sizeof(float));

  // Define work function for each thread
  auto threadFunction = [&](int startRow, int endRow) {
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
                  matrixB[k][j], matrixB[k][j + 1], matrixB[k][j + 2],
                  matrixB[k][j + 3], matrixB[k][j + 4], matrixB[k][j + 5],
                  matrixB[k][j + 6], matrixB[k][j + 7]);

              // Fused multiply-add: c_vals += a_val * b_vals
              c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
            }
          }

          // Store computed values back to result matrix
          _mm256_storeu_ps(&result[i][j], c_vals);
        }

        // Handle remaining columns (when N not divisible by 8)
        for (int j = (std::min(bj + BLOCK_SIZE, N) / 8 * 8);
             j < std::min(bj + BLOCK_SIZE, N); j++) {
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
    threads.push_back(std::thread(threadFunction, startRow, endRow));
  }

  // Wait for all threads to complete
  for (unsigned int t = 0; t < threads.size(); t++) {
    threads[t].join();
  }
}

#ifdef HAVE_PTHREAD
// Wrapper function for pthread-based implementation (uses native pthreads)
void multiplyMatricesPthreadWrapper(float **matrixA, float **matrixB,
                                    float **result) {
  multiplyMatricesPthreadAVX2(matrixA, matrixB, result, N, BLOCK_SIZE,
                              NUM_THREADS);
}
#endif

#ifdef HAVE_TBB
// Wrapper function for TBB-based implementation
void multiplyMatricesTBBWrapper(float **matrixA, float **matrixB,
                                float **result) {
  multiplyMatricesTBBAVX2(matrixA, matrixB, result, N, BLOCK_SIZE);
}
#endif
