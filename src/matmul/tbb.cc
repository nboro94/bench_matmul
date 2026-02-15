#include "tbb_matmul.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) &&                               \
    (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif
#include <oneapi/tbb.h>

// Parallel transpose using TBB
static void transposeMatrixTBB(float **input, float **output, int N,
                               int BLOCK_SIZE) {
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<int>(0, N, BLOCK_SIZE),
      [&](const oneapi::tbb::blocked_range<int> &range) {
        for (int bi = range.begin(); bi < range.end(); bi += BLOCK_SIZE) {
          for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int i = bi; i < std::min(bi + BLOCK_SIZE, N); i++) {
              for (int j = bj; j < std::min(bj + BLOCK_SIZE, N); j++) {
                output[j][i] = input[i][j];
              }
            }
          }
        }
      });
}

static inline float horizontalSumAVX2(__m256 v) {
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, v);
  float sum = 0.0f;
  for (int i = 0; i < 8; ++i) {
    sum += tmp[i];
  }
  return sum;
}

static inline float dotProductAVX2(const float *a, const float *b, int n) {
  __m256 acc = _mm256_setzero_ps();
  int k = 0;
  for (; k + 7 < n; k += 8) {
    const __m256 aVals = _mm256_loadu_ps(a + k);
    const __m256 bVals = _mm256_loadu_ps(b + k);
    acc = _mm256_fmadd_ps(aVals, bVals, acc);
  }
  float sum = horizontalSumAVX2(acc);
  for (; k < n; ++k) {
    sum += a[k] * b[k];
  }
  return sum;
}

#if defined(__AVX512F__)
static inline float horizontalSumAVX512(__m512 v) {
  alignas(64) float tmp[16];
  _mm512_store_ps(tmp, v);
  float sum = 0.0f;
  for (int i = 0; i < 16; ++i) {
    sum += tmp[i];
  }
  return sum;
}

static inline float dotProductAVX512(const float *a, const float *b, int n) {
  __m512 acc = _mm512_setzero_ps();
  int k = 0;
  for (; k + 15 < n; k += 16) {
    const __m512 aVals = _mm512_loadu_ps(a + k);
    const __m512 bVals = _mm512_loadu_ps(b + k);
    acc = _mm512_fmadd_ps(aVals, bVals, acc);
  }
  float sum = horizontalSumAVX512(acc);
  for (; k < n; ++k) {
    sum += a[k] * b[k];
  }
  return sum;
}

static bool cpuSupportsAVX512F() {
#if defined(_MSC_VER)
  int info[4] = {0, 0, 0, 0};
  __cpuid(info, 0);
  if (info[0] < 7) {
    return false;
  }

  __cpuid(info, 1);
  const bool osxsave = (info[2] & (1 << 27)) != 0;
  if (!osxsave) {
    return false;
  }

  const unsigned long long xcr0 = _xgetbv(0);
  // XMM (bit1), YMM (bit2), Opmask (bit5), ZMM_Hi256 (bit6), Hi16_ZMM (bit7)
  const unsigned long long required = (1ULL << 1) | (1ULL << 2) | (1ULL << 5) |
                                      (1ULL << 6) | (1ULL << 7);
  if ((xcr0 & required) != required) {
    return false;
  }

  __cpuidex(info, 7, 0);
  const bool avx512f = (info[1] & (1 << 16)) != 0;
  return avx512f;
#elif (defined(__GNUC__) || defined(__clang__)) &&                               \
    (defined(__x86_64__) || defined(__i386__))
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;

  if (__get_cpuid_max(0, nullptr) < 7) {
    return false;
  }

  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return false;
  }

  if ((ecx & bit_OSXSAVE) == 0) {
    return false;
  }

  unsigned int xcr0Low = 0;
  unsigned int xcr0High = 0;
  __asm__ volatile("xgetbv" : "=a"(xcr0Low), "=d"(xcr0High) : "c"(0));
  const unsigned long long xcr0 =
      (static_cast<unsigned long long>(xcr0High) << 32) | xcr0Low;

  // XMM (bit1), YMM (bit2), Opmask (bit5), ZMM_Hi256 (bit6), Hi16_ZMM (bit7)
  const unsigned long long required = (1ULL << 1) | (1ULL << 2) | (1ULL << 5) |
                                      (1ULL << 6) | (1ULL << 7);
  if ((xcr0 & required) != required) {
    return false;
  }

  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return false;
  }

  return (ebx & bit_AVX512F) != 0;
#else
  return false;
#endif
}
#endif

void multiplyMatricesTBBAVX2(float **matrixA, float **matrixB, float **result,
                             int N, int BLOCK_SIZE) {
  // Allocate transposed B matrix
  float *transposedData = new float[N * N];
  float **transposedB = new float *[N];
  for (int i = 0; i < N; i++) {
    transposedB[i] = &transposedData[i * N];
  }

  // Parallel transpose of matrix B using TBB
  transposeMatrixTBB(matrixB, transposedB, N, BLOCK_SIZE);

#if defined(__AVX512F__)
  const bool useAVX512 = cpuSupportsAVX512F();
#endif

  // Parallel matrix multiplication using TBB parallel_for
  // TBB automatically handles load balancing and thread management
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<int>(0, N),
      [&](const oneapi::tbb::blocked_range<int> &rowRange) {
        // Each task processes its assigned rows
        for (int i = rowRange.begin(); i < rowRange.end(); i++) {
          const float *aRow = matrixA[i];
          // Process in blocks for cache efficiency
          for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            const int jEnd = std::min(bj + BLOCK_SIZE, N);
            for (int j = bj; j < jEnd; ++j) {
#if defined(__AVX512F__)
              if (useAVX512) {
                result[i][j] = dotProductAVX512(aRow, transposedB[j], N);
              } else {
                result[i][j] = dotProductAVX2(aRow, transposedB[j], N);
              }
#else
              result[i][j] = dotProductAVX2(aRow, transposedB[j], N);
#endif
            }
          }
        }
      });

  // Cleanup
  delete[] transposedB;
  delete[] transposedData;
}
