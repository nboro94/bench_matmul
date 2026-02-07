#ifndef TBB_MATMUL_H
#define TBB_MATMUL_H

/**
 * Parallel-SIMD-AVX2 matrix multiplication using Intel TBB.
 * Combines parallel execution with AVX2 SIMD vectorization.
 * Uses TBB's parallel_for for automatic load balancing.
 *
 * @param matrixA First input matrix
 * @param matrixB Second input matrix
 * @param result Output matrix where result is stored
 * @param N Matrix dimension (N x N)
 * @param BLOCK_SIZE Block size for cache tiling
 */
void multiplyMatricesTBBAVX2(float **matrixA, float **matrixB, float **result,
                             int N, int BLOCK_SIZE);

#endif // TBB_MATMUL_H
