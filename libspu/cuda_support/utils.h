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

#pragma once

#include <cstddef>
#include <memory>

#include "libspu/core/shape.h"

namespace spu::cuda {

std::shared_ptr<std::byte> allocate(size_t bytes);

void deallocate(std::byte* data) noexcept;

void CopyToCudaDevice(const std::byte* src, size_t buf_size, const Shape& shape,
                      const Strides& strides, std::byte* dst, size_t elsize);

void DeinterleaveCopyToCudaDevice(const std::byte* src, size_t buf_size,
                                  const Shape& shape, const Strides& strides,
                                  std::byte* dst0, std::byte* dst1,
                                  size_t elsize);

void CopyFromCudaDevice(const std::byte* src, std::byte* dst, int64_t numel,
                        int64_t elsize, int64_t stride);

void printGPUData(const int* ptr, size_t numel);

bool hasGPUDevice();

}  // namespace spu::cuda
