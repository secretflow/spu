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

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

using PrgSeed = uint128_t;
using PrgCounter = uint64_t;

struct PrgArrayDesc {
  Shape shape;
  FieldType field;
  PrgCounter prg_counter;
};

inline NdArrayRef prgCreateArray(FieldType field, const Shape& shape,
                                 PrgSeed seed, PrgCounter* counter,
                                 PrgArrayDesc* desc) {
  if (desc != nullptr) {
    *desc = {Shape(shape.begin(), shape.end()), field, *counter};
  }
  return ring_rand(field, shape, seed, counter);
}

inline NdArrayRef prgReplayArray(PrgSeed seed, const PrgArrayDesc& desc) {
  PrgCounter counter = desc.prg_counter;
  return ring_rand(desc.field, desc.shape, seed, &counter);
}

}  // namespace spu::mpc
