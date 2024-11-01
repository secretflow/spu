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

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/UB/IR/UBOps.h"  // IWYU pragma: keep   PoisonAttr

#include "libspu/dialect/pphlo/IR/ops.h"

namespace mlir::spu::pphlo {

OpFoldResult ConstantOp::fold([[maybe_unused]] FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

OpFoldResult ConvertOp::fold(FoldAdaptor) {
  if (getOperand().getType() == getResult().getType()) {
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

  // reverse(reverse(x, dims), dims) = x
  if (auto prev = input.getDefiningOp<ReverseOp>()) {
    if (prev.getDimensions() == dims) {
      return prev.getOperand();
    }
  }

  return {};
}

OpFoldResult ReciprocalOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(),
                                     [](const APFloat& a) {
                                       APFloat one(a.getSemantics(), 1);
                                       return one / a;
                                     });
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

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  if (!adaptor.getLhs() || !adaptor.getRhs()) {
    return {};
  }

  if (isa<TypedAttr>(adaptor.getLhs()) && isa<TypedAttr>(adaptor.getRhs())) {
    auto lhs = cast<DenseIntOrFPElementsAttr>(adaptor.getLhs());
    auto rhs = cast<DenseIntOrFPElementsAttr>(adaptor.getRhs());

    if (lhs.getType() == rhs.getType()) {
      // int * int
      if (isa<IntegerType>(lhs.getElementType())) {
        return constFoldBinaryOp<IntegerAttr>(
            adaptor.getOperands(),
            [](const APInt& a, const APInt& b) { return a * b; });
      }
      // float * float
      if (isa<FloatType>(lhs.getElementType())) {
        return constFoldBinaryOp<FloatAttr>(
            adaptor.getOperands(),
            [](const APFloat& a, const APFloat& b) { return a * b; });
      }
    }

    // mixed type, currently only handle splat
    if (isa<SplatElementsAttr>(adaptor.getLhs()) &&
        isa<SplatElementsAttr>(adaptor.getRhs())) {
      // Both operands are splats so we can avoid expanding the values out and
      // just fold based on the splat value.
      auto lhs = cast<SplatElementsAttr>(adaptor.getLhs());
      auto rhs = cast<SplatElementsAttr>(adaptor.getRhs());

      auto calc = [](const APFloat& lhs, const APInt& rhs, bool rhs_is_signed) {
        APFloat rhs_f = APFloat(lhs.getSemantics());
        rhs_f.convertFromAPInt(rhs, rhs_is_signed,
                               APFloat::roundingMode::NearestTiesToEven);

        return rhs_f * lhs;
      };

      if (isa<FloatType>(lhs.getElementType()) &&
          isa<IntegerType>(rhs.getElementType())) {
        auto lhs_v = lhs.getSplatValue<APFloat>();
        auto rhs_v = rhs.getSplatValue<APInt>();
        auto rhs_isSigned =
            !(dyn_cast<IntegerType>(rhs.getElementType()).isUnsigned());

        auto elementResult = calc(lhs_v, rhs_v, rhs_isSigned);

        return DenseElementsAttr::get(cast<ShapedType>(lhs.getType()),
                                      elementResult);
      } else if (isa<IntegerType>(lhs.getElementType()) &&
                 isa<FloatType>(rhs.getElementType())) {
        auto lhs_v = lhs.getSplatValue<APInt>();
        auto rhs_v = rhs.getSplatValue<APFloat>();
        auto lhs_isSigned =
            !(dyn_cast<IntegerType>(lhs.getElementType()).isUnsigned());

        auto elementResult = calc(rhs_v, lhs_v, lhs_isSigned);

        return DenseElementsAttr::get(cast<ShapedType>(rhs.getType()),
                                      elementResult);
      }
    }
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

OpFoldResult SliceOp::fold(FoldAdaptor) {
  if (getOperand().getType() == getResult().getType()) {
    return getOperand();
  }
  return {};
}

}  // namespace mlir::spu::pphlo
