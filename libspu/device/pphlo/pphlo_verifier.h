// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/dialect/pphlo_dialect.h"
#include "libspu/dialect/pphlo_ops.h"
#include "libspu/dialect/pphlo_types.h"

namespace spu::device::pphlo {

class PPHloVerifier {
 private:
  SPUContext *ctx_{nullptr};
  mlir::MLIRContext mlir_ctx_;
  std::function<void(bool)> mismatch_handler_{[](bool) {}};

 public:
  explicit PPHloVerifier(SPUContext *ctx);

  void setMismatchHandler(std::function<void(bool)> f) {
    mismatch_handler_ = std::move(f);
  }

#define VERIFY_DECL(OpName)                                               \
  void verify(mlir::pphlo::OpName, absl::Span<const spu::Value> operands, \
              absl::Span<const spu::Value> expected);

  // Simple unary
  VERIFY_DECL(AbsOp)
  VERIFY_DECL(ReciprocalOp)
  VERIFY_DECL(NegOp)
  VERIFY_DECL(LogOp)
  VERIFY_DECL(Log1pOp)
  VERIFY_DECL(FloorOp)
  VERIFY_DECL(CeilOp)
  VERIFY_DECL(LogisticOp)
  VERIFY_DECL(TanhOp)
  VERIFY_DECL(NotOp)
  VERIFY_DECL(ExpOp)
  VERIFY_DECL(Expm1Op)
  VERIFY_DECL(RsqrtOp)
  VERIFY_DECL(SignOp)
  VERIFY_DECL(SqrtOp)
  VERIFY_DECL(RoundOp)

  // Simple binary
  VERIFY_DECL(AddOp)
  VERIFY_DECL(SubtractOp)
  VERIFY_DECL(MulOp)
  VERIFY_DECL(PowOp)
  VERIFY_DECL(MaxOp)
  VERIFY_DECL(MinOp)
  VERIFY_DECL(AndOp)
  VERIFY_DECL(OrOp)
  VERIFY_DECL(XorOp)
  VERIFY_DECL(DivOp)
  VERIFY_DECL(RemOp)
  VERIFY_DECL(DotOp)
  VERIFY_DECL(DotGeneralOp)

  // Comparison
  VERIFY_DECL(EqualOp)
  VERIFY_DECL(NotEqualOp)
  VERIFY_DECL(LessOp)
  VERIFY_DECL(LessEqualOp)
  VERIFY_DECL(GreaterOp)
  VERIFY_DECL(GreaterEqualOp)

  // Ternary
  VERIFY_DECL(SelectOp)
  VERIFY_DECL(ClampOp)

  // type conversion
  VERIFY_DECL(BitcastConvertOp)
  VERIFY_DECL(ConvertOp)

  // Conv
  VERIFY_DECL(ConvolutionOp)

  // Slice and update slice
  VERIFY_DECL(DynamicSliceOp)
  VERIFY_DECL(DynamicUpdateSliceOp)

  // Gather
  VERIFY_DECL(GatherOp)

  // Geometrical
  VERIFY_DECL(PadOp)
  VERIFY_DECL(BroadcastOp)
  VERIFY_DECL(ConcatenateOp)
  VERIFY_DECL(ReshapeOp)
  VERIFY_DECL(ReverseOp)
  VERIFY_DECL(SliceOp)
  VERIFY_DECL(TransposeOp)

  // Const
  VERIFY_DECL(IotaOp)

  // Reduce
  VERIFY_DECL(ReduceOp)
  VERIFY_DECL(ReduceWindowOp)

  // SelectAndScatter
  VERIFY_DECL(SelectAndScatterOp)

  // Shift
  VERIFY_DECL(ShiftLeftOp)
  VERIFY_DECL(ShiftRightArithmeticOp)
  VERIFY_DECL(ShiftRightLogicalOp)

  // Sort
  VERIFY_DECL(SortOp)

#undef VERIFY_DECL

// Other (no verify)
#define NO_VERIFY_DEFN(OpName)                                   \
  void verify(mlir::pphlo::OpName, absl::Span<const spu::Value>, \
              absl::Span<const spu::Value>) {}
  NO_VERIFY_DEFN(DbgPrintOp)
  NO_VERIFY_DEFN(IfOp)
  NO_VERIFY_DEFN(WhileOp)
  NO_VERIFY_DEFN(CaseOp)
  NO_VERIFY_DEFN(ReturnOp)
  NO_VERIFY_DEFN(RngOp)
  NO_VERIFY_DEFN(ConstantOp)
  NO_VERIFY_DEFN(MaxPoolScatterOp)
  NO_VERIFY_DEFN(PreferAOp)
  NO_VERIFY_DEFN(ArgMaxOp)
  NO_VERIFY_DEFN(EpsilonOp)

#undef NO_VERIFY_DEFN
};

}  // namespace spu::device::pphlo
