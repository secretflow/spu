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

#include <type_traits>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#include "spu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

template <typename HloOpTy>
struct HloToPPHloOpImpl {
  using Type = std::false_type;
};

template <typename HloOpTy>
using HloToPPHloOp = typename HloToPPHloOpImpl<HloOpTy>::Type;

#define MAP_HLO_TO_PPHLO(OpName)                                               \
  template <>                                                                  \
  struct HloToPPHloOpImpl<mhlo::OpName> {                                      \
    using Type = pphlo::OpName;                                                \
  };

#define MAP_HLO_TO_PPHLO_DIFF_NAME(HloName, PPHloName)                         \
  template <>                                                                  \
  struct HloToPPHloOpImpl<mhlo::HloName> {                                     \
    using Type = pphlo::PPHloName;                                             \
  };

MAP_HLO_TO_PPHLO(AbsOp)
MAP_HLO_TO_PPHLO(AddOp)
MAP_HLO_TO_PPHLO(AndOp)
MAP_HLO_TO_PPHLO(BitcastConvertOp)
MAP_HLO_TO_PPHLO(CeilOp)
MAP_HLO_TO_PPHLO(ClampOp)
MAP_HLO_TO_PPHLO(ConcatenateOp)
MAP_HLO_TO_PPHLO(ConstOp)
MAP_HLO_TO_PPHLO(ConvertOp)
MAP_HLO_TO_PPHLO(ConvOp)
MAP_HLO_TO_PPHLO(DynamicSliceOp)
MAP_HLO_TO_PPHLO(DynamicUpdateSliceOp)
MAP_HLO_TO_PPHLO(DivOp)
MAP_HLO_TO_PPHLO(DotOp)
MAP_HLO_TO_PPHLO(ExpOp)
MAP_HLO_TO_PPHLO(IotaOp)
MAP_HLO_TO_PPHLO(FloorOp)
MAP_HLO_TO_PPHLO(GatherOp)
MAP_HLO_TO_PPHLO(LogOp)
MAP_HLO_TO_PPHLO(Log1pOp)
MAP_HLO_TO_PPHLO(LogisticOp)
MAP_HLO_TO_PPHLO(MaxOp)
MAP_HLO_TO_PPHLO(MinOp)
MAP_HLO_TO_PPHLO(MulOp)
MAP_HLO_TO_PPHLO(NegOp)
MAP_HLO_TO_PPHLO(NotOp)
MAP_HLO_TO_PPHLO(OrOp)
MAP_HLO_TO_PPHLO(PadOp)
MAP_HLO_TO_PPHLO(PowOp)
MAP_HLO_TO_PPHLO(ReduceOp)
MAP_HLO_TO_PPHLO(ReduceWindowOp)
MAP_HLO_TO_PPHLO(RemOp)
MAP_HLO_TO_PPHLO(ReshapeOp)
MAP_HLO_TO_PPHLO(ReverseOp)
MAP_HLO_TO_PPHLO(RngUniformOp)
MAP_HLO_TO_PPHLO(SelectOp)
MAP_HLO_TO_PPHLO(ShiftLeftOp)
MAP_HLO_TO_PPHLO(ShiftRightArithmeticOp)
MAP_HLO_TO_PPHLO(ShiftRightLogicalOp)
MAP_HLO_TO_PPHLO(SliceOp)
MAP_HLO_TO_PPHLO(SortOp)
MAP_HLO_TO_PPHLO(SqrtOp)
MAP_HLO_TO_PPHLO(SubOp)
MAP_HLO_TO_PPHLO(TanhOp)
MAP_HLO_TO_PPHLO(TransposeOp)
MAP_HLO_TO_PPHLO(XorOp)

MAP_HLO_TO_PPHLO_DIFF_NAME(BroadcastInDimOp, BroadcastOp)

#undef MAP_HLO_TO_PPHLO

} // namespace mlir::pphlo
