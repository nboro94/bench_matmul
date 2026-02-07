#include "pthread_matmul.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>


// Worker function for matrix multiplication - processes assigned rows
static void matmulWorker(float **matrixA, float **transposedB, float **result,
                         int N, int BLOCK_SIZE, int startRow, int endRow) {
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
}

// Transpose worker for parallel transposition
static void transposeWorker(float **input, float **output, int N,
                            int BLOCK_SIZE, int startBlock, int endBlock) {
  for (int bi = startBlock; bi < endBlock; bi += BLOCK_SIZE) {
    for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
      for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
        for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
          output[j][i] = input[i][j];
        }
      }
    }
  }
}

void multiplyMatricesPthreadAVX2(float **matrixA, float **matrixB,
                                 float **result, int N, int BLOCK_SIZE,
                                 int numThreads) {
  // Allocate transposed B matrix
  float *transposedData = new float[N * N];
  float **transposedB = new float *[N];
  for (int i = 0; i < N; i++) {
    transposedB[i] = &transposedData[i * N];
  }

  // Parallel transpose of matrix B using std::thread
  std::vector<std::thread> transposeThreads;
  int blocksPerThread = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / numThreads;
  blocksPerThread = std::max(1, blocksPerThread) * BLOCK_SIZE;

  for (int t = 0; t < numThreads; t++) {
    int startBlock = t * blocksPerThread;
    int endBlock = (t == numThreads - 1) ? N : (t + 1) * blocksPerThread;
    transposeThreads.emplace_back(transposeWorker, matrixB, transposedB, N,
                                  BLOCK_SIZE, startBlock, endBlock);
  }

  for (auto &th : transposeThreads) {
    th.join();
  }

  // Initialize result matrix to zeros
  memset(result[0], 0, N * N * sizeof(float));

  // Create worker threads for matrix multiplication using std::thread
  std::vector<std::thread> threads;
  int rowsPerThread = N / numThreads;

  for (int t = 0; t < numThreads; t++) {
    int startRow = t * rowsPerThread;
    int endRow = (t == numThreads - 1) ? N : (t + 1) * rowsPerThread;
    threads.emplace_back(matmulWorker, matrixA, transposedB, result, N,
                         BLOCK_SIZE, startRow, endRow);
  }

  // Wait for all threads to complete
  for (auto &th : threads) {
    th.join();
  }

  // Cleanup
  delete[] transposedB;
  delete[] transposedData;
}
