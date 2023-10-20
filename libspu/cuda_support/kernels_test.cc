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

#include "libspu/cuda_support/kernels.h"

#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/cuda_support/utils.h"

namespace spu::cuda {

template <typename T>
std::shared_ptr<std::byte> GenerateGPUData(int64_t numel, T start = 0) {
  auto gpu_ptr = allocate(numel * sizeof(T));

  std::vector<T> data(numel);
  std::iota(data.begin(), data.end(), start);

  CopyToCudaDevice((std::byte*)data.data(), numel * sizeof(T), {numel}, {1},
                   gpu_ptr.get(), sizeof(T));

  return gpu_ptr;
}

template <typename T>
std::vector<T> getGPUData(std::byte* gpu_ptr, int64_t numel) {
  std::vector<T> cpu(numel);
  CopyFromCudaDevice(gpu_ptr, (std::byte*)cpu.data(), numel, sizeof(T), 1);
  return cpu;
}

TEST(CudaKernels, BasicAdd) {
  auto x = GenerateGPUData<uint64_t>(10);
  auto y = GenerateGPUData<uint64_t>(10, 10);

  add(reinterpret_cast<uint64_t*>(x.get()),
      reinterpret_cast<uint64_t*>(y.get()), 10);

  auto ret = getGPUData<uint64_t>(x.get(), 10);

  for (size_t idx = 0; idx < 10; ++idx) {
    EXPECT_EQ(ret[idx], idx + 10 + idx);
  }
}

TEST(CudaKernels, Matmul) {
  auto x = GenerateGPUData<uint64_t>(12, 1);  // 4x3
  auto y = GenerateGPUData<uint64_t>(6, 1);   // 3x2
  auto c = allocate(8 * sizeof(uint64_t));

  matmul(4, 2, 3, reinterpret_cast<uint64_t*>(x.get()),
         reinterpret_cast<uint64_t*>(y.get()),
         reinterpret_cast<uint64_t*>(c.get()));

  auto ret = getGPUData<uint64_t>(c.get(), 8);

  std::vector<uint64_t> expected = {22, 28, 49, 64, 76, 100, 103, 136};
  for (size_t idx = 0; idx < 8; ++idx) {
    EXPECT_EQ(ret[idx], expected[idx]);
  }
}

}  // namespace spu::cuda
