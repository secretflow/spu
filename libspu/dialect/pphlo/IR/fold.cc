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
#include "mlir/IR/Matchers.h"

#include "libspu/dialect/pphlo/IR/ops.h"

namespace mlir::spu::pphlo {

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

OpFoldResult TruncOp::fold(FoldAdaptor adaptor) {
  if (getOperand().getType() == getResult().getType()) {
    return getOperand();
  }
  // Assume y = convert(x), z = trunc(y), y must be a FXP value and y must have
  // more FXP bits than z. So if z has the same type as x, x must be a FXP value
  // and have more FXP bits than y. The convert is actually an untrunc op. So
  // for z = trunc(untrunc(x)), we can just return x.
  if (auto operand = getOperand().getDefiningOp<ConvertOp>()) {
    if (operand.getOperand().getType() == getResult().getType()) {
      return operand.getOperand();
    }
  }

  // If trunc a constant, forward
  return adaptor.getOperand();
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

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  if (APInt rhsVal; matchPattern(adaptor.getRhs(), m_ConstantInt(&rhsVal))) {
    /// or(x, 0) -> x
    if (rhsVal.isZero()) {
      return getLhs();
    }
    /// or(x, <all ones>) -> <all ones>
    if (rhsVal.isAllOnes()) {
      return adaptor.getRhs();
    }
  }

  APInt intValue;
  /// or(x, xor(x, 1)) -> 1
  if (matchPattern(getRhs(), m_Op<XorOp>(matchers::m_Val(getLhs()),
                                         m_ConstantInt(&intValue))) &&
      intValue.isAllOnes()) {
    return getRhs().getDefiningOp<XorOp>().getRhs();
  }
  /// or(xor(x, 1), x) -> 1
  if (matchPattern(getLhs(), m_Op<XorOp>(matchers::m_Val(getRhs()),
                                         m_ConstantInt(&intValue))) &&
      intValue.isAllOnes()) {
    return getLhs().getDefiningOp<XorOp>().getRhs();
  }

  return {};
}

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  /// and(x, 0) -> 0
  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    return getRhs();
  }
  /// and(x, allOnes) -> x
  // FIXME: for 1bit value, runtime x value might have more than 1 actual bit.
  // So it's not very safe to do fold to x.
  APInt intValue;
  if (matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue)) &&
      intValue.getBitWidth() > 1 && intValue.isAllOnes()) {
    return getLhs();
  }
  /// and(x, not(x)) -> 0
  if (matchPattern(getRhs(), m_Op<XorOp>(matchers::m_Val(getLhs()),
                                         m_ConstantInt(&intValue))) &&
      intValue.isAllOnes()) {
    return Builder(getContext()).getZeroAttr(getType());
  }

  return {};
}

OpFoldResult XorOp::fold(FoldAdaptor adaptor) {
  /// xor(x, 0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    return getLhs();
  }
  /// xor(x, x) -> 0
  if (getLhs() == getRhs()) {
    return Builder(getContext()).getZeroAttr(getType());
  }
  /// xor(xor(x, a), a) -> x
  /// xor(xor(a, x), a) -> x
  if (auto prev = getLhs().getDefiningOp<XorOp>()) {
    if (prev.getRhs() == getRhs()) {
      return prev.getLhs();
    }
    if (prev.getLhs() == getRhs()) {
      return prev.getRhs();
    }
  }
  /// xor(a, xor(x, a)) -> x
  /// xor(a, xor(a, x)) -> x
  if (auto prev = getRhs().getDefiningOp<XorOp>()) {
    if (prev.getRhs() == getLhs()) {
      return prev.getLhs();
    }
    if (prev.getLhs() == getLhs()) {
      return prev.getRhs();
    }
  }

  return {};
}

OpFoldResult BitcastConvertOp::fold(FoldAdaptor adaptor) {
  if (getType() == getOperand().getType()) {
    return getOperand();
  }

  TypeTools tools(getContext());

  if (tools.isIntType(getType()) && tools.isIntType(getOperand().getType())) {
    return constFoldCastOp<IntegerAttr, IntegerAttr>(
        adaptor.getOperand(), getType(),
        [](const APInt& in, bool&) { return in; });
  }

  return {};
}

}  // namespace mlir::spu::pphlo
