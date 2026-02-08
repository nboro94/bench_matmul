#include "matmul.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <sstream>

namespace {

constexpr int kBlockDim = 16;

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
  if (status == cudaSuccess) {
    return true;
  }
  std::ostringstream os;
  os << "CUDA error at " << step << ": " << cudaGetErrorString(status);
  log(os.str());
  return false;
}

} // namespace

void multiplyMatricesCUDA(float **matrixA, float **matrixB, float **result) {
  if (matrixA == nullptr || matrixB == nullptr || result == nullptr) {
    log("CUDA: received null matrix pointer");
    return;
  }

  int n = N;
  size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) *
                 sizeof(float);

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  if (!checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(A)") ||
      !checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(B)") ||
      !checkCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc(C)")) {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return;
  }

  if (!checkCuda(cudaMemcpy(deviceA, matrixA[0], bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(A)") ||
      !checkCuda(cudaMemcpy(deviceB, matrixB[0], bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(B)")) {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return;
  }

  dim3 block(kBlockDim, kBlockDim);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);

  if (!checkCuda(cudaGetLastError(), "kernel launch") ||
      !checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return;
  }

  if (!checkCuda(cudaMemcpy(result[0], deviceC, bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(C)")) {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return;
  }

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}
