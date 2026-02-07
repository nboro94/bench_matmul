#ifndef MATMUL_H
#define MATMUL_H

#ifdef HAVE_TBB
#include "tbb_matmul.h"
#endif

// Naive implementations
void multiplyMatrices(float **matrixA, float **matrixB, float **result);
void multiplyMatricesNaiveParallel(float **matrixA, float **matrixB,
                                   float **result);

// Block/Tiled implementations
void multiplyMatricesOptimized(float **matrixA, float **matrixB,
                               float **result);
void multiplyMatricesOptimizedParallel(float **matrixA, float **matrixB,
                                       float **result);

// Transposed implementations
void multiplyMatricesTransposed(float **matrixA, float **matrixB,
                                float **result);
void multiplyMatricesLocalTranspose(float **matrixA, float **matrixB,
                                    float **result);

// SIMD implementations
void multiplyMatricesAVX2(float **matrixA, float **matrixB, float **result);
void multiplyMatricesAVX2NoTranspose(float **matrixA, float **matrixB,
                                     float **result);
void multiplyMatricesThreaded(float **matrixA, float **matrixB, float **result);
void multiplyMatricesThreadedAVX2NoTranspose(float **matrixA, float **matrixB,
                                             float **result);

// Scalar Unrolled implementations
void multiplyMatricesOptimizedNoSIMD(float **matrixA, float **matrixB,
                                     float **result);
void multiplyMatricesOptimizedNoSIMDThreaded(float **matrixA, float **matrixB,
                                             float **result);

// Pthread wrapper
void multiplyMatricesPthreadWrapper(float **matrixA, float **matrixB,
                                    float **result);

#ifdef HAVE_TBB
// TBB wrapper
void multiplyMatricesTBBWrapper(float **matrixA, float **matrixB,
                                float **result);
#endif

#endif // MATMUL_H
