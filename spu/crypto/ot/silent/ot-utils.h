// Copyright 2021 Ant Group Co., Ltd.
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

#include "emp-tool/utils/block.h"

namespace spu {
template <typename basetype>
void pack_ot_messages(basetype *y, const basetype *const *data,
                      const emp::block *pad, size_t ysize, size_t bsize,
                      size_t bitsize, size_t N);

template <typename basetype>
void unpack_ot_messages(basetype *data, const uint8_t *r, const basetype *recvd,
                        const emp::block *pad, size_t bsize, size_t bitsize,
                        size_t N);

void pack_cot_messages(uint64_t *y, const uint64_t *corr_data, size_t ysize,
                       size_t bsize, size_t bitsize);

void unpack_cot_messages(uint64_t *corr_data, const uint64_t *recvd,
                         size_t bsize, size_t bitsize);

}  // namespace spu
