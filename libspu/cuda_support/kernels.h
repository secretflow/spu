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

#include <cstdint>

namespace spu::cuda {

// uint64 implementation
void matmul(int64_t M, int64_t N, int64_t K, const uint64_t* A, uint64_t* B,
            uint64_t* C);

void add(uint64_t* A, const uint64_t* B, int64_t numel);

}  // namespace spu::cuda
