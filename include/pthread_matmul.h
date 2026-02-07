#ifndef PTHREAD_MATMUL_H
#define PTHREAD_MATMUL_H

/**
 * Parallel-SIMD-AVX2 matrix multiplication using POSIX pthreads.
 * Combines parallel execution with AVX2 SIMD vectorization.
 *
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 * @param N Matrix dimension (N x N)
 * @param BLOCK_SIZE Block size for cache tiling
 * @param numThreads Number of threads to use
 */
void multiplyMatricesPthreadAVX2(float **matrixA, float **matrixB,
                                 float **result, int N, int BLOCK_SIZE,
                                 int numThreads);

#endif // PTHREAD_MATMUL_H
