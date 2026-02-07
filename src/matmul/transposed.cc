#include "matmul.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <vector>

void multiplyMatricesTransposed(float **matrixA, float **matrixB,
                                float **result) {
  // Create transposed copy of B for better cache locality
  float **transposedB = allocateMatrix();
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

void multiplyMatricesLocalTranspose(float **matrixA, float **matrixB,
                                    float **result) {
  // Initialize result matrix to zeros
  memset(result[0], 0, N * N * sizeof(float));

  // Allocate aligned buffer dynamically for block transposition
  std::vector<float> localTransposedB(BLOCK_SIZE * BLOCK_SIZE);
  float *localTransposedBPtr = localTransposedB.data();

  // Process matrices in blocks
  for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
    for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
      // For each block in result matrix, process corresponding blocks in inputs
      for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
        // Transpose current block of B into local buffer
        for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
          for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
            localTransposedBPtr[(k - bk) * BLOCK_SIZE + (j - bj)] =
                matrixB[k][j];
          }
        }

        // Process current block with transposed data
        for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
          for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
            // Accumulate dot product for current element
            float sum = result[i][j];

            // Access transposed data from local buffer for better cache
            // behavior
            for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
              sum += matrixA[i][k] *
                     localTransposedBPtr[(k - bk) * BLOCK_SIZE + (j - bj)];
            }

            result[i][j] = sum;
          }
        }
      }
    }
  }
}
