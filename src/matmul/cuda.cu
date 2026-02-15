#include "benchmark.h"
#include "matmul.h"
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <sstream>

namespace {

constexpr int kBlockDim = 16;
#if defined(USE_DEBUG_MODE)
constexpr bool use_debug_mode = (USE_DEBUG_MODE != 0);
#else
constexpr bool use_debug_mode = false;
#endif

inline void debugLog(const char *message) {
  if (use_debug_mode) {
    log(message);
  }
}

std::string cudaVersionToString(int rawVersion) {
  int major = rawVersion / 1000;
  int minor = (rawVersion % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

bool checkCudaDriverCompatibility() {
  CUresult initStatus = cuInit(0);
  if (initStatus != CUDA_SUCCESS) {
    const char *errName = nullptr;
    const char *errString = nullptr;
    cuGetErrorName(initStatus, &errName);
    cuGetErrorString(initStatus, &errString);
    std::ostringstream os;
    os << "CUDA driver init failed";
    if (errName != nullptr) {
      os << " (" << errName << ")";
    }
    if (errString != nullptr) {
      os << ": " << errString;
    }
    GLOBAL_BENCHMARK_KERNEL_FAILURE = true;
    GLOBAL_BENCHMARK_FAILURE_REASON = os.str();
    log(GLOBAL_BENCHMARK_FAILURE_REASON);
    return false;
  }

  int driverVersion = 0;
  CUresult drvStatus = cuDriverGetVersion(&driverVersion);
  if (drvStatus != CUDA_SUCCESS) {
    const char *errName = nullptr;
    const char *errString = nullptr;
    cuGetErrorName(drvStatus, &errName);
    cuGetErrorString(drvStatus, &errString);
    std::ostringstream os;
    os << "Unable to query CUDA driver version";
    if (errName != nullptr) {
      os << " (" << errName << ")";
    }
    if (errString != nullptr) {
      os << ": " << errString;
    }
    GLOBAL_BENCHMARK_KERNEL_FAILURE = true;
    GLOBAL_BENCHMARK_FAILURE_REASON = os.str();
    log(GLOBAL_BENCHMARK_FAILURE_REASON);
    return false;
  }

  if (driverVersion < CUDART_VERSION) {
    std::ostringstream os;
    os << "CUDA driver/runtime mismatch: driver supports CUDA "
       << cudaVersionToString(driverVersion) << ", but benchmark was built "
       << "with CUDA runtime " << cudaVersionToString(CUDART_VERSION)
       << ". Update GPU driver or build with an older CUDA toolkit.";
    GLOBAL_BENCHMARK_KERNEL_FAILURE = true;
    GLOBAL_BENCHMARK_FAILURE_REASON = os.str();
    log(GLOBAL_BENCHMARK_FAILURE_REASON);
    return false;
  }

  return true;
}

__global__ void matmulKernel(const float *a, const float *b, float *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;
    int rowOffset = row * n;
    for (int k = 0; k < n; ++k) {
      sum += a[rowOffset + k] * b[k * n + col];
    }
    c[rowOffset + col] = sum;
  }
}

bool checkCuda(cudaError_t status, const char *step) {
  if (status != cudaSuccess) {
    std::ostringstream os;
    os << "CUDA error at " << step << ": " << cudaGetErrorString(status);
    GLOBAL_BENCHMARK_KERNEL_FAILURE = true;
    GLOBAL_BENCHMARK_FAILURE_REASON = os.str();
    log(GLOBAL_BENCHMARK_FAILURE_REASON);
    return false;
  }
  return true;
}

} // namespace

void multiplyMatricesCUDA(float **matrixA, float **matrixB, float **result) {
  debugLog("CUDA: entered multiplyMatricesCUDA");
  if (matrixA == nullptr || matrixB == nullptr || result == nullptr) {
    GLOBAL_BENCHMARK_KERNEL_FAILURE = true;
    GLOBAL_BENCHMARK_FAILURE_REASON = "CUDA: received null matrix pointer";
    log(GLOBAL_BENCHMARK_FAILURE_REASON);
    return;
  }

  int n = N;
  size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) *
                 sizeof(float);
  debugLog("CUDA: checking driver/runtime compatibility");
  if (!checkCudaDriverCompatibility()) {
    return;
  }

  debugLog("CUDA: probing device availability");
  int deviceCount = 0;
  if (!checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) {
    return;
  }
  debugLog("CUDA: host-side memset of result");
  memset(result[0], 0, bytes);
  if (deviceCount <= 0) {
    GLOBAL_BENCHMARK_KERNEL_FAILURE = true;
    GLOBAL_BENCHMARK_FAILURE_REASON = "CUDA error: no CUDA-capable device found";
    log(GLOBAL_BENCHMARK_FAILURE_REASON);
    return;
  }
  if (!checkCuda(cudaSetDevice(0), "cudaSetDevice(0)")) {
    return;
  }

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;
  auto cleanup = [&]() {
    debugLog("CUDA: cleanup");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
  };

  debugLog("CUDA: cudaMalloc A/B/C");
  if (!checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(A)") ||
      !checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(B)") ||
      !checkCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc(C)")) {
    cleanup();
    return;
  }

  debugLog("CUDA: cudaMemcpy A/B");
  if (!checkCuda(cudaMemcpy(deviceA, matrixA[0], bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(A)") ||
      !checkCuda(cudaMemcpy(deviceB, matrixB[0], bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(B)")) {
    cleanup();
    return;
  }

  dim3 block(kBlockDim, kBlockDim);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  debugLog("CUDA: launching kernel");
  matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);

  if (!checkCuda(cudaGetLastError(), "kernel launch") ||
      !checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    cleanup();
    return;
  }

  if (!checkCuda(cudaMemcpy(result[0], deviceC, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(C)")) {
    cleanup();
    return;
  }

  cleanup();
}
