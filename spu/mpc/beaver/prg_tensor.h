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

#include "spu/core/array_ref.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {

using PrgSeed = uint128_t;
using PrgCounter = uint64_t;

struct PrgArrayDesc {
  size_t numel;
  FieldType field;
  PrgCounter prg_counter;
};

inline ArrayRef prgCreateArray(FieldType field, size_t size, PrgSeed seed,
                               PrgCounter* counter, PrgArrayDesc* desc) {
  *desc = {size, field, *counter};
  return ring_rand(field, size, seed, counter);
}

inline ArrayRef prgReplayArray(PrgSeed seed, const PrgArrayDesc& desc) {
  PrgCounter counter = desc.prg_counter;
  return ring_rand(desc.field, desc.numel, seed, &counter);
}

}  // namespace spu::mpc
