#include "matmul.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

void multiplyMatricesOptimizedNoSIMD(float **matrixA, float **matrixB,
                                     float **result) {
  // Create transposed copy of B for better memory access patterns
  float **transposedB = allocateMatrix();
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
              sum += matrixA[i][k + 1] * transposedB[j][k + 1];
              sum += matrixA[i][k + 2] * transposedB[j][k + 2];
              sum += matrixA[i][k + 3] * transposedB[j][k + 3];
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

void multiplyMatricesOptimizedNoSIMDThreaded(float **matrixA, float **matrixB,
                                             float **result) {
  // Create transposed copy of B for better memory access patterns
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
              sum += matrixA[i][k + 1] * transposedB[j][k + 1];
              sum += matrixA[i][k + 2] * transposedB[j][k + 2];
              sum += matrixA[i][k + 3] * transposedB[j][k + 3];
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
    threads.push_back(std::thread(threadFunction, startRow, endRow));
  }

  // Wait for all threads to complete
  for (unsigned int t = 0; t < threads.size(); t++) {
    threads[t].join();
  }

  // Release temporary transposed matrix
  deallocateMatrix(transposedB);
}
