// Copyright 2023 zhangwfjh
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

#include <string>
#include <vector>

#include "yacl/base/buffer.h"

namespace spu::psi {

namespace {

thread_local std::vector<yacl::Buffer> buffers;

}  // namespace

namespace Buffer {

inline void Push(yacl::Buffer&& buffer) {
  buffers.push_back(std::move(buffer));
}

inline yacl::Buffer Merge() {
  auto count = buffers.size();
  auto stride = buffers.front().size();
  yacl::Buffer buffer(count * stride);
  for (size_t i{}; i != count; ++i) {
    std::memcpy(buffer.data<char>() + i * stride, buffers[i].data(), stride);
  }
  buffers.clear();
  return buffer;
}

inline void Split(yacl::Buffer&& buffer, size_t count = 1) {
  auto stride = buffer.size() / count;
  buffers.assign(count, yacl::Buffer(stride));
  for (size_t i{}; i != count; ++i) {
    std::memcpy(buffers[count - 1 - i].data(), buffer.data<char>() + i * stride,
                stride);
  }
}

inline yacl::Buffer Pop() {
  auto buf = buffers.back();
  buffers.pop_back();
  return buf;
}

};  // namespace Buffer

}  // namespace spu::psi
