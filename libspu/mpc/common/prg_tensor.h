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

#include "libspu/core/memref.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

using PrgSeed = uint128_t;
using PrgCounter = uint64_t;

struct PrgArrayDesc {
  Shape shape;
  size_t field;
  PrgCounter prg_counter;
};

inline void prgCreateArray(MemRef& in, PrgSeed seed, PrgCounter* counter,
                           PrgArrayDesc* desc) {
  if (desc != nullptr) {
    auto shape = in.shape();
    size_t field = in.eltype().as<BaseRingType>()->valid_bits();
    *desc = {Shape(shape.begin(), shape.end()), field, *counter};
  }
  ring_rand(in, seed, counter);
}

inline MemRef prgReplayArray(PrgSeed seed, const PrgArrayDesc& desc) {
  PrgCounter counter = desc.prg_counter;
  MemRef in(makeType<RingTy>(GetSemanticType(desc.field), desc.field),
            desc.shape);
  ring_rand(in, seed, &counter);
  return in;
}

inline MemRef prgReplayArrayMutable(PrgSeed seed, PrgArrayDesc& desc) {
  MemRef in(makeType<RingTy>(GetSemanticType(desc.field), desc.field),
            desc.shape);
  ring_rand(in, seed, &desc.prg_counter);
  return in;
}

}  // namespace spu::mpc
