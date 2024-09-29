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
#include <type_traits>

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/ring/IR/ops.h"

namespace mlir::spu {

template <typename HloOpTy>
struct PPHloToRingOpDirectImpl {
  using Type = std::false_type;
};

template <typename PPHloOpTy>
using PPHloToRingOpDirect = typename PPHloToRingOpDirectImpl<PPHloOpTy>::Type;

#define DIRECT_MAP_PPHLO_TO_RING(OpName)          \
  template <>                                     \
  struct PPHloToRingOpDirectImpl<pphlo::OpName> { \
    using Type = ring::OpName;                    \
  };

DIRECT_MAP_PPHLO_TO_RING(AddOp)
DIRECT_MAP_PPHLO_TO_RING(AndOp)
DIRECT_MAP_PPHLO_TO_RING(BitRevOp)
DIRECT_MAP_PPHLO_TO_RING(DotOp)
DIRECT_MAP_PPHLO_TO_RING(EqualOp)
DIRECT_MAP_PPHLO_TO_RING(LessOp)
DIRECT_MAP_PPHLO_TO_RING(MulOp)
DIRECT_MAP_PPHLO_TO_RING(NegOp)
DIRECT_MAP_PPHLO_TO_RING(NotOp)
DIRECT_MAP_PPHLO_TO_RING(XorOp)

#undef DIRECT_MAP_PPHLO_TO_RING

}  // namespace mlir::spu
