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

#include "libspu/cuda_support/utils.h"

#include <array>
#include <numeric>

#include "gtest/gtest.h"

namespace spu::cuda {

TEST(Memory, Allocate) {
  auto gpu_ptr = allocate(10);

  EXPECT_NE(gpu_ptr, nullptr);
}

TEST(Memory, TestCompactCopy1D) {
  auto gpu_ptr = allocate(10 * sizeof(int));

  std::array<int, 10> base;
  std::iota(base.begin(), base.end(), 0);

  CopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()), 10 * sizeof(int),
                   {10}, {1}, gpu_ptr.get(), sizeof(int));

  std::array<int, 10> from_gpu;
  CopyFromCudaDevice(gpu_ptr.get(),
                     reinterpret_cast<std::byte*>(from_gpu.data()), 10,
                     sizeof(int), 1);

  for (size_t idx = 0; idx < 10; ++idx) {
    EXPECT_EQ(base[idx], from_gpu[idx]);
  }
}

TEST(Memory, TestCompactDeinterleaveCopy1D) {
  auto gpu_ptr1 = allocate(10 * sizeof(int));
  auto gpu_ptr2 = allocate(10 * sizeof(int));

  std::array<int, 20> base;
  std::iota(base.begin(), base.end(), 0);

  DeinterleaveCopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()),
                               20 * sizeof(int), {10}, {1}, gpu_ptr1.get(),
                               gpu_ptr2.get(), 2 * sizeof(int));

  std::array<int, 10> from_gpu1;
  std::array<int, 10> from_gpu2;
  CopyFromCudaDevice(gpu_ptr1.get(),
                     reinterpret_cast<std::byte*>(from_gpu1.data()), 10,
                     sizeof(int), 1);
  CopyFromCudaDevice(gpu_ptr2.get(),
                     reinterpret_cast<std::byte*>(from_gpu2.data()), 10,
                     sizeof(int), 1);

  for (size_t idx = 0; idx < 10; ++idx) {
    EXPECT_EQ(base[2 * idx], from_gpu1[idx]);
    EXPECT_EQ(base[2 * idx + 1], from_gpu2[idx]);
  }
}

TEST(Memory, TestCompactCopy2D) {
  auto gpu_ptr = allocate(7 * 2304 * sizeof(int));

  Shape s{7, 2304};
  std::array<int, 7 * 2304> base;
  std::iota(base.begin(), base.end(), 0);

  CopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()),
                   7 * 2304 * sizeof(int), s, {2304, 1}, gpu_ptr.get(),
                   sizeof(int));

  std::array<int, 7 * 2304> from_gpu;
  CopyFromCudaDevice(gpu_ptr.get(),
                     reinterpret_cast<std::byte*>(from_gpu.data()), 7 * 2304,
                     sizeof(int), 1);

  for (size_t idx = 0; idx < 7 * 2304; ++idx) {
    EXPECT_EQ(base[idx], from_gpu[idx]);
  }
}

TEST(Memory, TestCompactDeinterleaveCopy2D) {
  auto gpu_ptr1 = allocate(25 * sizeof(int));
  auto gpu_ptr2 = allocate(25 * sizeof(int));

  Shape s{5, 5};
  std::array<int, 50> base;
  std::iota(base.begin(), base.end(), 0);

  DeinterleaveCopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()),
                               50 * sizeof(int), s, {5, 1}, gpu_ptr1.get(),
                               gpu_ptr2.get(), 2 * sizeof(int));

  std::array<int, 25> from_gpu1;
  std::array<int, 25> from_gpu2;
  CopyFromCudaDevice(gpu_ptr1.get(),
                     reinterpret_cast<std::byte*>(from_gpu1.data()), 25,
                     sizeof(int), 1);
  CopyFromCudaDevice(gpu_ptr2.get(),
                     reinterpret_cast<std::byte*>(from_gpu2.data()), 25,
                     sizeof(int), 1);

  for (size_t idx = 0; idx < 25; ++idx) {
    EXPECT_EQ(base[2 * idx], from_gpu1[idx]);
    EXPECT_EQ(base[2 * idx + 1], from_gpu2[idx]);
  }
}

TEST(Memory, TestStridedCopy1D) {
  auto gpu_ptr = allocate(10 * sizeof(int));

  std::array<int, 20> base;
  std::iota(base.begin(), base.end(), 0);

  CopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()), 20 * sizeof(int),
                   {10}, {2}, gpu_ptr.get(), sizeof(int));

  std::array<int, 10> from_gpu;
  CopyFromCudaDevice(gpu_ptr.get(),
                     reinterpret_cast<std::byte*>(from_gpu.data()), 10,
                     sizeof(int), 1);

  for (size_t idx = 0; idx < 10; ++idx) {
    EXPECT_EQ(base[2 * idx], from_gpu[idx]);
  }
}

TEST(Memory, TestStridedDeinterleaveCopy1D) {
  auto gpu_ptr1 = allocate(10 * sizeof(int));
  auto gpu_ptr2 = allocate(10 * sizeof(int));

  std::array<int, 40> base;
  std::iota(base.begin(), base.end(), 0);

  DeinterleaveCopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()),
                               40 * sizeof(int), {10}, {2}, gpu_ptr1.get(),
                               gpu_ptr2.get(), 2 * sizeof(int));

  std::array<int, 10> from_gpu1;
  std::array<int, 10> from_gpu2;

  CopyFromCudaDevice(gpu_ptr1.get(),
                     reinterpret_cast<std::byte*>(from_gpu1.data()), 10,
                     sizeof(int), 1);
  CopyFromCudaDevice(gpu_ptr2.get(),
                     reinterpret_cast<std::byte*>(from_gpu2.data()), 10,
                     sizeof(int), 1);

  for (size_t idx = 0; idx < 10; ++idx) {
    EXPECT_EQ(base[4 * idx], from_gpu1[idx]);
    EXPECT_EQ(base[4 * idx + 1], from_gpu2[idx]);
  }
}

TEST(Memory, TestStridedCopy2D) {
  auto gpu_ptr = allocate(12 * sizeof(int));

  Shape shape = {3, 4};
  Strides strides = {8, 2};

  std::array<int, 24> base;
  std::iota(base.begin(), base.end(), 0);

  CopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()), 24 * sizeof(int),
                   shape, strides, gpu_ptr.get(), sizeof(int));

  std::array<int, 12> from_gpu;
  CopyFromCudaDevice(gpu_ptr.get(),
                     reinterpret_cast<std::byte*>(from_gpu.data()), 12,
                     sizeof(int), 1);

  for (size_t idx = 0; idx < 12; ++idx) {
    EXPECT_EQ(base[2 * idx], from_gpu[idx]);
  }
}

TEST(Memory, TestStridedCopy2D2) {
  auto gpu_ptr = allocate(12 * sizeof(int));

  Shape shape = {3, 4};
  Strides strides = {16, 2};

  std::array<int, 48> base;
  std::iota(base.begin(), base.end(), 0);

  CopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()), 48 * sizeof(int),
                   shape, strides, gpu_ptr.get(), sizeof(int));

  std::array<int, 12> from_gpu;
  CopyFromCudaDevice(gpu_ptr.get(),
                     reinterpret_cast<std::byte*>(from_gpu.data()), 12,
                     sizeof(int), 1);

  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      EXPECT_EQ(base[row * strides[0] + col * strides[1]],
                from_gpu[row * shape[1] + col]);
    }
  }
}

TEST(Memory, TestStidedCopyFrom) {
  auto gpu_ptr = allocate(10 * sizeof(int));

  std::array<int, 10> base;
  std::iota(base.begin(), base.end(), 0);

  CopyToCudaDevice(reinterpret_cast<std::byte*>(base.data()), 10 * sizeof(int),
                   {10}, {1}, gpu_ptr.get(), sizeof(int));

  std::array<int, 20> from_gpu;
  CopyFromCudaDevice(gpu_ptr.get(),
                     reinterpret_cast<std::byte*>(from_gpu.data()), 10,
                     sizeof(int), 2);

  for (size_t idx = 0; idx < 10; ++idx) {
    EXPECT_EQ(base[idx], from_gpu[2 * idx]);
  }
}

}  // namespace spu::cuda
