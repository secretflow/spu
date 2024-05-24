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

#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

OpFoldResult ConstantOp::fold([[maybe_unused]] FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

OpFoldResult ConvertOp::fold(FoldAdaptor) {
  auto operand_ty = mlir::dyn_cast<TensorType>(getOperand().getType());
  auto result_ty = mlir::dyn_cast<TensorType>(getResult().getType());
  if (operand_ty == result_ty) {
    return getOperand();
  }

  return {};
}

OpFoldResult ReverseOp::fold(FoldAdaptor) {
  auto input = getOperand();

  // No dimensions to reverse.
  auto dims = getDimensions();
  if (dims.empty()) {
    return input;
  }

  // If the dimensions to reverse are all statically 1, then the reverse is a
  // no-op.
  auto shapedType = mlir::dyn_cast<ShapedType>(input.getType());
  if (llvm::all_of(
          dims, [&](int64_t dim) { return shapedType.getDimSize(dim) == 1; })) {
    return input;
  }
  return {};
}

OpFoldResult ReciprocalOp::fold(FoldAdaptor operands) {
  auto val =
      mlir::dyn_cast_or_null<DenseFPElementsAttr>(operands.getOperands()[0]);

  if (!val) {
    return {};
  }

  if (val.isSplat()) {
    auto splat_val = val.getSplatValue<APFloat>();
    APFloat one(splat_val.getSemantics(), 1);

    return SplatElementsAttr::get(mlir::dyn_cast<ShapedType>(val.getType()),
                                  one / splat_val);
  }

  llvm::SmallVector<APFloat, 4> values;
  values.reserve(val.getNumElements());

  auto first_val = *val.getValues<APFloat>().begin();
  APFloat one(first_val.getSemantics(), 1);

  for (auto it : val.getValues<APFloat>()) {
    values.push_back(one / it);
  }

  return DenseFPElementsAttr::get(mlir::dyn_cast<ShapedType>(val.getType()),
                                  values);
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) {
  auto operand_shape =
      mlir::dyn_cast<TensorType>(getOperand().getType()).getShape();
  auto result_shape =
      mlir::dyn_cast<TensorType>(getResult().getType()).getShape();
  if (operand_shape == result_shape) {
    return getOperand();
  }
  return {};
}

OpFoldResult TransposeOp::fold(FoldAdaptor) {
  for (const auto& it : llvm::enumerate(getPermutation())) {
    if (static_cast<int64_t>(it.index()) != it.value()) {
      return {};
    }
  }
  return getOperand();
}

}  // namespace mlir::spu::pphlo