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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "libspu/dialect/ring/IR/ops.h"

namespace mlir::spu::ring {

class ImproveMulSmallConstant : public OpRewritePattern<MulOp> {
 private:
  static int64_t findTwoK(double in) {
    uint64_t N = 1;
    int64_t count = 0;
    while (N < in) {
      N <<= 1;
      ++count;
    }
    return --count;
  }

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    // Find small constant
    Value const_v;
    Value non_const_v;
    int64_t fxp_bits = 0;
    Type fxp_type;

    if (auto lhs_to_fxp = op.getLhs().getDefiningOp<ring::EncodeToFxpOp>()) {
      const_v = lhs_to_fxp.getOperand();
      non_const_v = op.getRhs();
      fxp_bits = lhs_to_fxp.getFxpBits();
      fxp_type = lhs_to_fxp.getType();
    } else if (auto rhs_to_fxp =
                   op.getRhs().getDefiningOp<ring::EncodeToFxpOp>()) {
      const_v = rhs_to_fxp.getOperand();
      non_const_v = op.getLhs();
      fxp_bits = rhs_to_fxp.getFxpBits();
      fxp_type = rhs_to_fxp.getType();
    } else {
      return failure();
    }

    auto const_op = const_v.getDefiningOp<arith::ConstantOp>();
    if (const_op == nullptr) {
      return failure();
    }

    auto const_attr = cast<DenseElementsAttr>(const_op.getValue());

    // Not splat, ignore
    if (!const_attr.isSplat()) {
      return failure();
    }

    // Not float, ignore
    if (!mlir::isa<FloatType>(const_attr.getElementType())) {
      return failure();
    }

    auto apfloat_v = const_attr.getSplatValue<APFloat>();
    auto v = std::abs(apfloat_v.convertToDouble());
    bool is_negative = apfloat_v.convertToDouble() < 0;
    double eps = 1.0 / std::pow(2, fxp_bits);

    if (v <= 4 * eps) {
      // Is small value
      // Handle x * (very_small_const)
      // return truncate(x * n/N, k); n = 2^k
      // Compute N -> 1/fValue
      auto N = 1.0 / v;
      auto k = findTwoK(N);
      auto n = std::pow(2, k);

      // n/N
      APFloat nN(apfloat_v.getSemantics());

      if (&apfloat_v.getSemantics() == &llvm::APFloatBase::IEEEdouble()) {
        nN = APFloat(static_cast<double>(n / N));
      } else if (&apfloat_v.getSemantics() ==
                 &llvm::APFloatBase::IEEEsingle()) {
        nN = APFloat(static_cast<float>(n / N));
      } else {
        // Do not what type it is....
        return failure();
      }

      if (is_negative) {
        nN = -nN;
      }

      Value c0 = rewriter.create<arith::ConstantOp>(
          op->getLoc(), DenseFPElementsAttr::get(const_attr.getType(), nN));

      c0 = rewriter.create<ring::EncodeToFxpOp>(op->getLoc(), fxp_type, c0,
                                                fxp_bits);

      // x*n/N
      auto mul =
          rewriter.create<MulOp>(op->getLoc(), op.getType(), non_const_v, c0);

      // truncate(x*n/N, k)
      rewriter.replaceOpWithNewOp<TruncOp>(op, op.getType(), mul, k);

      return success();
    }

    return failure();
  }
};

class MergeConsecutiveTrunc : public OpRewritePattern<TruncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncOp op,
                                PatternRewriter& rewriter) const override {
    if (auto parent_trunc = op.getOperand().getDefiningOp<TruncOp>()) {
      // Merge two consecutive truncation
      auto parent_trunc_bits = parent_trunc.getBits();

      rewriter.replaceOpWithNewOp<TruncOp>(op, op.getType(),
                                           parent_trunc.getOperand(),
                                           parent_trunc_bits + op.getBits());
      return success();
    }

    return failure();
  }
};

class LessToMsb : public OpRewritePattern<LessOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LessOp op,
                                PatternRewriter& rewriter) const override {
    auto rhs = op.getRhs();

    if (auto cst = rhs.getDefiningOp<arith::ConstantOp>()) {
      if (matchPattern(cst.getValue(), m_Zero())) {
        auto cst_type = mlir::cast<ShapedType>(cst.getType());
        auto cst_el_type = mlir::cast<IntegerType>(cst_type.getElementType());

        if (cst_el_type.isUnsigned()) {
          // less(x, 0) for unsigned is constant false
          auto c = rewriter.create<arith::ConstantOp>(
              op->getLoc(),
              SplatElementsAttr::get(cst_type.clone(rewriter.getI1Type()),
                                     rewriter.getBoolAttr(false)));
          if (isPublic(op.getType())) {
            rewriter.replaceOp(op, c);
            return success();
          } else {
            rewriter.replaceOpWithNewOp<ring::P2SOp>(op, op.getType(), c);
            return success();
          }
        } else {
          /// less(x, 0) -> msb(x)
          rewriter.replaceOpWithNewOp<ring::MsbOp>(op, op.getLhs());
          return success();
        }
      }
    }

    return failure();
  }
};

#include "libspu/dialect/ring/IR/canonicalization_patterns.cc.inc"

void MulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context) {
  results.add<ImproveMulSmallConstant>(context);
}

void TruncOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                          ::mlir::MLIRContext* context) {
  results.add<MergeConsecutiveTrunc>(context);
}

void LessOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<LessToMsb>(context);
}

}  // namespace mlir::spu::ring
