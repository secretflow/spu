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

namespace mlir::spu::ring {

OpFoldResult CastOp::fold(FoldAdaptor) {
  if (getOperand().getType() == getResult().getType()) {
    return getOperand();
  }
  return {};
}

OpFoldResult TruncOp::fold(FoldAdaptor) {
  if (getBits() == 0) {
    return getOperand();
  }

  return {};
}

OpFoldResult EncodeToFxpOp::fold(FoldAdaptor) {
  if (auto decode = getOperand().getDefiningOp<DecodeFromFxpOp>()) {
    if (decode.getOperand().getType() == getType()) {
      return decode.getOperand();
    }
  }

  return {};
}

OpFoldResult DecodeFromFxpOp::fold(FoldAdaptor) {
  if (auto encode = getOperand().getDefiningOp<EncodeToFxpOp>()) {
    if (encode.getOperand().getType() == getType()) {
      return encode.getOperand();
    }
  }

  return {};
}

}  // namespace mlir::spu::ring