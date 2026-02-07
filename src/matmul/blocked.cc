#include "matmul.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

void multiplyMatricesOptimized(float **matrixA, float **matrixB,
                               float **result) {
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

void multiplyMatricesOptimizedParallel(float **matrixA, float **matrixB,
                                       float **result) {
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
              float *resRow = result[i];
              float *brow = matrixB[k];
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
  int blocksPerThread =
      std::max(1, (totalBlockRows + (int)NUM_THREADS - 1) / (int)NUM_THREADS);

  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);

  for (unsigned int t = 0; t < NUM_THREADS; ++t) {
    int startBlockIdx = t * blocksPerThread;
    int endBlockIdx = std::min(totalBlockRows, startBlockIdx + blocksPerThread);
    if (startBlockIdx >= endBlockIdx)
      break; // no more work

    int startBi = startBlockIdx * BLOCK_SIZE;
    int endBi = endBlockIdx * BLOCK_SIZE;
    if (startBi >= N)
      break;
    endBi = std::min(endBi, N);

    threads.emplace_back(worker, startBi, endBi);
  }

  for (auto &th : threads)
    th.join();
}
