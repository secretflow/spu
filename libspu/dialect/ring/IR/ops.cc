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

#include "libspu/dialect/ring/IR/ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

#include "libspu/dialect/ring/IR/ops.h.inc"
#include "libspu/dialect/utils/utils.h"

#define GET_OP_CLASSES
#include "libspu/dialect/ring/IR/ops.cc.inc"

namespace mlir::spu::ring {

void LShiftOp::build(OpBuilder& builder, OperationState& result, Value lhs,
                     IntegerAttr rhs) {
  auto lhs_type = mlir::cast<ShapedType>(lhs.getType());

  Value new_rhs;
  if (ShapedType::isDynamicShape(lhs_type.getShape())) {
    new_rhs = splatifyConstant(builder, rhs, lhs);
  } else {
    new_rhs = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        SplatElementsAttr::get(
            RankedTensorType::get(lhs_type.getShape(), rhs.getType()), rhs));
  }

  auto p_rhs_type =
      RankedTensorType::get(lhs_type.getShape(), getBaseType(lhs.getType()));
  new_rhs = builder.create<tensor::BitcastOp>(builder.getUnknownLoc(),
                                              p_rhs_type, new_rhs);

  build(builder, result, lhs_type, lhs, new_rhs);
}

void ARShiftOp::build(OpBuilder& builder, OperationState& result, Value lhs,
                      IntegerAttr rhs) {
  auto lhs_type = mlir::cast<ShapedType>(lhs.getType());

  Value new_rhs;
  if (ShapedType::isDynamicShape(lhs_type.getShape())) {
    new_rhs = splatifyConstant(builder, rhs, lhs);
  } else {
    new_rhs = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        SplatElementsAttr::get(
            RankedTensorType::get(lhs_type.getShape(), rhs.getType()), rhs));
  }

  auto p_rhs_type =
      RankedTensorType::get(lhs_type.getShape(), getBaseType(lhs.getType()));
  new_rhs = builder.create<tensor::BitcastOp>(builder.getUnknownLoc(),
                                              p_rhs_type, new_rhs);

  build(builder, result, lhs_type, lhs, new_rhs);
}

void RShiftOP::build(OpBuilder& builder, OperationState& result, Value lhs,
                     IntegerAttr rhs) {
  auto lhs_type = mlir::cast<ShapedType>(lhs.getType());

  Value new_rhs;
  if (ShapedType::isDynamicShape(lhs_type.getShape())) {
    new_rhs = splatifyConstant(builder, rhs, lhs);
  } else {
    new_rhs = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        SplatElementsAttr::get(
            RankedTensorType::get(lhs_type.getShape(), rhs.getType()), rhs));
  }

  auto p_rhs_type =
      RankedTensorType::get(lhs_type.getShape(), getBaseType(lhs.getType()));
  new_rhs = builder.create<tensor::BitcastOp>(builder.getUnknownLoc(),
                                              p_rhs_type, new_rhs);

  build(builder, result, lhs_type, lhs, new_rhs);
}

}  // namespace mlir::spu::ring