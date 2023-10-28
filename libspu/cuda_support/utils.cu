// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <string_view>

#include "libspu/cuda_support/utils.h"

namespace spu::cuda {

namespace kernels {

__global__ void printGPUData_(const int* gpu_ptr, size_t numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("print idx = %d", idx);
  for (size_t idx = 0; idx < numel; ++idx) {
    printf("%d ", gpu_ptr[idx]);
  }
  printf("\n");
}

__global__ void compactGPUMemory(std::byte* strided, std::byte* compact,
                                 int64_t col, int64_t s_row, int64_t s_col,
                                 int64_t elsize, int64_t numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numel) {
    return;
  }

  auto r = idx / col;
  auto c = idx % col;
  auto s_idx = r * s_row + c * s_col;
  s_idx *= elsize;
  for (int offset = 0; offset < elsize; ++offset) {
    compact[idx * elsize + offset] = strided[s_idx + offset];
  }
}

__global__ void deinterleaveCopy(std::byte* strided_interleave,
                                 std::byte* compact_p1, std::byte* compact_p2,
                                 int64_t col, int64_t s_row, int64_t s_col,
                                 int64_t elsize, int numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numel) {
    return;
  }

  auto r = idx / col;
  auto c = idx % col;

  auto s_idx = r * s_row + c * s_col;
  for (int offset = 0; offset < elsize; ++offset) {
    compact_p1[idx * elsize + offset] = strided_interleave[s_idx + offset];
    compact_p2[idx * elsize + offset] =
        strided_interleave[s_idx + offset + elsize];
  }
}

}  // namespace kernels

void checkGPUStatus(cudaError_t status, std::string_view msg) {
  if (status != cudaSuccess) {
    printf("%s %s", msg.data(), cudaGetErrorString(status));
  }
}

void printGPUData(const int* ptr, size_t numel) {
  dim3 block(1);
  dim3 grid(1);
  kernels::printGPUData_<<<grid, block>>>(ptr, numel);
}

std::shared_ptr<std::byte> allocate(size_t bytes) {
  std::byte* ptr = nullptr;
  cudaError_t error = cudaMalloc(&ptr, bytes);
  checkGPUStatus(error, "Failed to allocate GPU memory:");
  return std::shared_ptr<std::byte>(ptr, deallocate);
}

void deallocate(std::byte* ptr) noexcept {
  auto error = cudaFree(ptr);
  checkGPUStatus(error, "Failed to free GPU memory:");
}

void CopyToCudaDevice(const std::byte* src, size_t buf_size, const Shape& shape,
                      const Strides& strides, std::byte* dst, size_t elsize) {
#ifdef LOG_GPU_DATA_COPY
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(stop);
#endif

  auto numel = shape.numel();
  cudaError_t error;

  if (numel * elsize == buf_size) {
    // Straight copy
    error = cudaMemcpy(dst, src, buf_size, cudaMemcpyHostToDevice);
  } else {
    std::byte* tmp = nullptr;
    error = cudaMalloc(&tmp, buf_size);
    error = cudaMemcpy(tmp, src, buf_size, cudaMemcpyHostToDevice);

    // we allow max 1024 threads per block, and then scale out the copy across
    // multiple blocks
    dim3 block(std::min<int64_t>(numel, 1024));
    dim3 grid(numel / block.x + (numel % block.x == 0 ? 0 : 1));
    if (shape.ndim() == 1) {
      kernels::compactGPUMemory<<<grid, block>>>(tmp, dst, shape[0], 0,
                                                 strides[0], elsize, numel);
    } else {
      kernels::compactGPUMemory<<<grid, block>>>(tmp, dst, shape[1], strides[0],
                                                 strides[1], elsize, numel);
    }
    error = cudaFree(tmp);
  }

  checkGPUStatus(error, "Failed to copy to GPU: ");

#ifdef LOG_GPU_DATA_COPY
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Copy %ld bytes from %p takes %fms\n", (long)shape.numel() * elsize,
         src, milliseconds);
#endif
}

void DeinterleaveCopyToCudaDevice(const std::byte* src, size_t buf_size,
                                  const Shape& shape, const Strides& strides,
                                  std::byte* dst0, std::byte* dst1,
                                  size_t elsize) {
#ifdef LOG_GPU_DATA_COPY
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(stop);
#endif

  auto numel = shape.numel();
  cudaError_t error;

  // First copy to a strided gpu buffer
  std::byte* s_gpu_buffer = nullptr;
  error = cudaMalloc(&s_gpu_buffer, buf_size);
  error = cudaMemcpy(s_gpu_buffer, src, buf_size, cudaMemcpyHostToDevice);

  // Deinterleave
  dim3 block(std::min<int64_t>(numel, 1024));
  dim3 grid(numel / block.x + (numel % block.x == 0 ? 0 : 1));

  if (shape.ndim() == 1) {
    kernels::deinterleaveCopy<<<grid, block>>>(s_gpu_buffer, dst0, dst1,
                                               shape[0], 0, strides[0] * elsize,
                                               elsize / 2, numel);
  } else {
    kernels::deinterleaveCopy<<<grid, block>>>(
        s_gpu_buffer, dst0, dst1, shape[1], strides[0] * elsize,
        strides[1] * elsize, elsize / 2, numel);
  }

  error = cudaFree(s_gpu_buffer);
  checkGPUStatus(error, "Failed to copy to GPU and deinterleave data: ");

#ifdef LOG_GPU_DATA_COPY
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Copy %ld bytes from %p takes %fms\n", (long)shape.numel() * elsize,
         src, milliseconds);
#endif
}

void CopyFromCudaDevice(const std::byte* src, std::byte* dst, int64_t numel,
                        int64_t elsize, int64_t stride) {
  cudaError_t result;
  if (stride == 1) {
    result = cudaMemcpy(dst, src, numel * elsize, cudaMemcpyDeviceToHost);
  } else {
    result = cudaMemcpy2D(dst, stride * elsize, src, elsize, elsize, numel,
                          cudaMemcpyDeviceToHost);
  }
  checkGPUStatus(result, "Failed to copy to host: ");
}

bool initGPUState() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    if (prop.major >= 0) {
      printf("Use GPU:\n");
      printf("  Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      cudaSetDevice(i);
      return true;
    }
  }

  return false;
}

static bool hasGPU = false;
static std::once_flag flag;

bool hasGPUDevice() {
  std::call_once(flag, []() { hasGPU = initGPUState(); });
  return hasGPU;
}

}  // namespace spu::cuda
