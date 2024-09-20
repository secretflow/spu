// Copyright 2024 Ant Group Co., Ltd.
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

#include <cstdint>

namespace spu::mpc::semi2k::beaver::ttp_server {

constexpr size_t kUpStreamChunkSize = 50 * 1024 * 1024;    // bytes
constexpr size_t kDownStreamChunkSize = 50 * 1024 * 1024;  // bytes

struct BeaverPermUpStreamMeta {
  uint64_t total_buf_size;
};

// A list of buffer streams
struct BeaverDownStreamMeta {
  uint32_t total_buf_num;  // total buffer stream num
  int32_t err_code = 0;
};

}  // namespace spu::mpc::semi2k::beaver::ttp_server