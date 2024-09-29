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
#include "libspu/dialect/ring/IR/type_helper.h"

namespace mlir::spu::ring {

LogicalResult EncodeToFxpOp::verify() {
  int64_t bits = getFxpBits();
  int64_t width = getRingWidth(getType());

  if (bits > width) {
    return emitOpError("encoding fixedpoint fraction bits is invalid");
  }

  return success();
}

LogicalResult DecodeFromFxpOp::verify() {
  int64_t bits = getFxpBits();
  int64_t width = getRingWidth(getOperand().getType());

  if (bits > width) {
    return emitOpError("decoding fixedpoint fraction bits is invalid");
  }

  return success();
}

LogicalResult TruncOp::verify() {
  int64_t bits = getBits();
  int64_t width = getRingWidth(getOperand().getType());

  if (bits >= width) {
    return emitOpError("truncate bits is invalid");
  }

  return success();
}

LogicalResult DotOp::verify() {
  auto lhsShape = mlir::cast<ShapedType>(getLhs().getType()).getShape();
  auto rhsShape = mlir::cast<ShapedType>(getRhs().getType()).getShape();
  auto retShape = mlir::cast<ShapedType>(getType()).getShape();

  if (lhsShape.size() != 2 || (lhsShape.size() != rhsShape.size())) {
    return emitOpError("expect 2D dot 2D");
  }

  if (lhsShape[0] != retShape[0] && rhsShape[1] != retShape[1]) {
    return emitOpError("inputs & result shape mismatch");
  }

  if (lhsShape[1] != rhsShape[0]) {
    return emitOpError("contracting dimension mismatch");
  }

  return success();
}

LogicalResult SecretInsertSliceOp::verify() {
  // Make sure there must be at least one secret indices
  for (auto indice : getStartIndices()) {
    if (isSecret(indice.getType())) {
      return success();
    }
  }

  return failure();
}

}  // namespace mlir::spu::ring
