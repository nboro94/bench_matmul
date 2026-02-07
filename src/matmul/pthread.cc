#ifdef HAVE_PTHREAD
#include "pthread_matmul.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <pthread.h>
#include <vector>


struct ThreadArgs {
  float **matrixA;
  float **transposedB;
  float **result;
  int N;
  int BLOCK_SIZE;
  int startRow;
  int endRow;
};

// Worker function for matrix multiplication
static void *matmulWorker(void *arg) {
  ThreadArgs *args = (ThreadArgs *)arg;
  float **matrixA = args->matrixA;
  float **transposedB = args->transposedB;
  float **result = args->result;
  int N = args->N;
  int BLOCK_SIZE = args->BLOCK_SIZE;
  int startRow = args->startRow;
  int endRow = args->endRow;

  for (int i = startRow; i < endRow; i++) {
    for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
      for (int j = bj; j < std::min(bj + BLOCK_SIZE, N - 7); j += 8) {
        __m256 c_vals = _mm256_setzero_ps();
        for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
          for (int k = bk; k < std::min(bk + BLOCK_SIZE, N); k++) {
            __m256 a_val = _mm256_set1_ps(matrixA[i][k]);
            __m256 b_vals =
                _mm256_setr_ps(transposedB[j][k], transposedB[j + 1][k],
                               transposedB[j + 2][k], transposedB[j + 3][k],
                               transposedB[j + 4][k], transposedB[j + 5][k],
                               transposedB[j + 6][k], transposedB[j + 7][k]);
            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
          }
        }
        _mm256_storeu_ps(&result[i][j], c_vals);
      }
      // Handle remaining columns
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
  return NULL;
}

struct TransposeArgs {
  float **input;
  float **output;
  int N;
  int BLOCK_SIZE;
  int startBlock;
  int endBlock;
};

static void *transposeWorker(void *arg) {
  TransposeArgs *args = (TransposeArgs *)arg;
  float **input = args->input;
  float **output = args->output;
  int N = args->N;
  int BLOCK_SIZE = args->BLOCK_SIZE;
  int startBlock = args->startBlock;
  int endBlock = args->endBlock;

  for (int bi = startBlock; bi < endBlock; bi += BLOCK_SIZE) {
    for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
      for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
        for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
          output[j][i] = input[i][j];
        }
      }
    }
  }
  return NULL;
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

  // Parallel transpose using pthreads
  pthread_t *transposeThreads = new pthread_t[numThreads];
  TransposeArgs *transposeArgs = new TransposeArgs[numThreads];
  int blocksPerThread = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / numThreads;
  blocksPerThread = std::max(1, blocksPerThread) * BLOCK_SIZE;

  for (int t = 0; t < numThreads; t++) {
    int startBlock = t * blocksPerThread;
    int endBlock = (t == numThreads - 1) ? N : (t + 1) * blocksPerThread;
    transposeArgs[t] = {matrixB,    transposedB, N,
                        BLOCK_SIZE, startBlock,  endBlock};
    if (pthread_create(&transposeThreads[t], NULL, transposeWorker,
                       &transposeArgs[t]) != 0) {
      std::cerr << "Error creating transpose thread " << t << std::endl;
    }
  }

  for (int t = 0; t < numThreads; t++) {
    pthread_join(transposeThreads[t], NULL);
  }
  delete[] transposeThreads;
  delete[] transposeArgs;

  // Initialize result matrix
  memset(result[0], 0, N * N * sizeof(float));

  // Matrix multiplication using pthreads
  pthread_t *workerThreads = new pthread_t[numThreads];
  ThreadArgs *workerArgs = new ThreadArgs[numThreads];
  int rowsPerThread = N / numThreads;

  for (int t = 0; t < numThreads; t++) {
    int startRow = t * rowsPerThread;
    int endRow = (t == numThreads - 1) ? N : (t + 1) * rowsPerThread;
    workerArgs[t] = {matrixA,    transposedB, result, N,
                     BLOCK_SIZE, startRow,    endRow};
    if (pthread_create(&workerThreads[t], NULL, matmulWorker, &workerArgs[t]) !=
        0) {
      std::cerr << "Error creating worker thread " << t << std::endl;
    }
  }

  for (int t = 0; t < numThreads; t++) {
    pthread_join(workerThreads[t], NULL);
  }
  delete[] workerThreads;
  delete[] workerArgs;

  // Cleanup
  delete[] transposedB;
  delete[] transposedData;
}
#endif // HAVE_PTHREAD
