#include "matmul.h"
#include "utils.h"
#include <cstring>
#include <thread>
#include <vector>

void multiplyMatrices(float **matrixA, float **matrixB, float **result) {
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

void multiplyMatricesNaiveParallel(float **matrixA, float **matrixB,
                                   float **result) {
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
    if (start >= end)
      break;
    threadPool.emplace_back(worker, start, end);
  }

  for (auto &th : threadPool)
    th.join();
}
