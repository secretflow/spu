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

#include <numeric>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/core/prelude.h"
#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {

namespace {

#include "libspu/dialect/pphlo/transforms/decompose_patterns.cc.inc"

struct PowerDecompose : public OpRewritePattern<PowOp> {
 public:
  explicit PowerDecompose(MLIRContext *context)
      : OpRewritePattern<PowOp>(context) {}

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tool(op->getContext());

    auto result_t = op.getResult().getType();
    bool result_is_int = tool.isIntType(result_t);

    if (!result_is_int) {
      return failure();
    }

    // Int power is very rare and has strange behavior, leave it as an intrinsic
    rewriter.replaceOpWithNewOp<CustomCallOp>(op, op->getResultTypes(),
                                              op->getOperands(), IPOW);

    return success();
  }
};

struct IDivDecompose : public OpRewritePattern<DivOp> {
 public:
  explicit IDivDecompose(MLIRContext *context)
      : OpRewritePattern<DivOp>(context) {}

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tool(op->getContext());

    auto base_t =
        mlir::dyn_cast<IntegerType>(tool.getBaseType(op.getResult().getType()));

    if (!base_t) {
      return failure();
    }

    bool is_uint = base_t.isUnsigned();

    auto x = op.getLhs();
    auto y = op.getRhs();

    Value sign_x;
    Value sign_y;
    Value abs_x;
    Value abs_y;

    if (!is_uint) {
      sign_x = rewriter.create<SignOp>(op->getLoc(), x, /*ignore_zero*/ true);
      sign_y = rewriter.create<SignOp>(op->getLoc(), y, /*ignore_zero*/ true);

      abs_x = rewriter.create<MulOp>(op->getLoc(), x, sign_x);
      abs_y = rewriter.create<MulOp>(op->getLoc(), y, sign_y);
    } else {
      abs_x = x;
      abs_y = y;
    }

    Value q;
    {
      auto ft_type = tool.promoteToFloatType(op.getResult().getType());
      auto x_f = rewriter.create<ConvertOp>(op->getLoc(), ft_type, abs_x);
      auto y_f = rewriter.create<ConvertOp>(op->getLoc(), ft_type, abs_y);

      Value approx_q = rewriter.create<DivOp>(op->getLoc(), x_f, y_f);

      // Due to truncation error and limited precision of fxp, the approximate
      // quotient should be corrected
      approx_q = rewriter.create<ConvertOp>(op->getLoc(),
                                            op.getResult().getType(), approx_q);

      auto approx_x = rewriter.create<MulOp>(op->getLoc(), abs_y, approx_q);

      // if (approx_q + 1) * y <= x, then ++approx_q;
      Value v1 = rewriter.create<LessEqualOp>(
          op->getLoc(), rewriter.create<AddOp>(op->getLoc(), approx_x, abs_y),
          abs_x);
      // if approx_q * y > x, then --approx_q;
      Value v2 = rewriter.create<GreaterOp>(op->getLoc(), approx_x, abs_x);

      // Fix type
      v1 = rewriter.create<ConvertOp>(op->getLoc(), approx_q.getType(), v1);
      v2 = rewriter.create<ConvertOp>(op->getLoc(), approx_q.getType(), v2);

      q = rewriter.create<SubtractOp>(
          op->getLoc(), rewriter.create<AddOp>(op->getLoc(), approx_q, v1), v2);
    }

    Value ret;
    if (is_uint) {
      ret = q;
    } else {
      ret = rewriter.create<MulOp>(
          op->getLoc(), q,
          rewriter.create<MulOp>(op->getLoc(), sign_x, sign_y));
    }

    rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(), ret);

    return success();
  }
};

struct CeilDecompose : public OpRewritePattern<CeilOp> {
 public:
  explicit CeilDecompose(MLIRContext *context)
      : OpRewritePattern<CeilOp>(context) {}

  LogicalResult matchAndRewrite(CeilOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tool(op->getContext());
    auto el_type = tool.getBaseType(op.getOperand().getType());
    auto in_shape_type = mlir::dyn_cast<ShapedType>(
        tool.getExpressedType(op.getOperand().getType()));
    Attribute k1_v = rewriter.getFloatAttr(el_type, 1.0);

    // ceil(x) = floor(x + 1.0 - epsilon)
    auto k1 = rewriter.create<arith::ConstantOp>(
        op->getLoc(), DenseElementsAttr::get(in_shape_type, k1_v));

    Value x = rewriter.create<arith::SubFOp>(
        op->getLoc(), k1,
        rewriter.create<EpsilonOp>(op->getLoc(), in_shape_type));
    x = rewriter.create<AddOp>(op->getLoc(), op.getOperand(), x);

    rewriter.replaceOpWithNewOp<FloorOp>(op, x);
    return success();
  }
};

struct RoundDecompose : public OpRewritePattern<RoundOp> {
 public:
  explicit RoundDecompose(MLIRContext *context)
      : OpRewritePattern<RoundOp>(context) {}

  LogicalResult matchAndRewrite(RoundOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tool(op->getContext());
    auto el_type = tool.getBaseType(op.getOperand().getType());
    auto in_shape_type = mlir::dyn_cast<ShapedType>(
        tool.getExpressedType(op.getOperand().getType()));
    Attribute half_v = rewriter.getFloatAttr(el_type, 0.5);

    // select(x < 0, (int)(x-0.5), (int)(x+0.5))
    // -> (float)(int)(x + sign(x) * 0.5)
    Value p_half = rewriter.create<arith::ConstantOp>(
        op->getLoc(), DenseElementsAttr::get(in_shape_type, half_v));

    auto sign = rewriter.create<SignOp>(op->getLoc(), op.getOperand(),
                                        /*ignore_zero*/ true);

    p_half = rewriter.create<MulOp>(op->getLoc(), sign, p_half);

    auto rounded =
        rewriter.create<AddOp>(op->getLoc(), op.getOperand(), p_half);

    // cast to int
    auto int_type = tool.demoteTypeToInt(op.getOperand().getType());
    auto casted = rewriter.create<ConvertOp>(op->getLoc(), int_type, rounded);

    // cast back to float
    rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(), casted);

    return success();
  }
};

struct RemainderDecompose : public OpRewritePattern<RemOp> {
 public:
  explicit RemainderDecompose(MLIRContext *context)
      : OpRewritePattern<RemOp>(context) {}

  LogicalResult matchAndRewrite(RemOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tool(op->getContext());
    // lhs/rhs
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    // 1st: find quotient by x/y
    Value quotient = rewriter.create<DivOp>(op->getLoc(), lhs, rhs);

    if (tool.isFloatType(lhs.getType()) || tool.isFloatType(rhs.getType())) {
      // 2nd: round to nearest number through (x >= 0.0) ? floor(x) : ceil(x)...
      auto zero_v =
          rewriter.getZeroAttr(tool.getExpressedType(quotient.getType()));
      auto zero = rewriter.create<arith::ConstantOp>(op->getLoc(), zero_v);

      auto pred = rewriter.create<GreaterEqualOp>(op->getLoc(), quotient, zero);
      auto floor = rewriter.create<FloorOp>(op->getLoc(), quotient);
      auto ceil = rewriter.create<CeilOp>(op->getLoc(), quotient);
      quotient = rewriter.create<SelectOp>(op->getLoc(), floor.getType(), pred,
                                           floor, ceil);
    }

    // 3rd: rem = numer - rquot * denom
    rewriter.replaceOpWithNewOp<SubtractOp>(
        op, lhs, rewriter.create<MulOp>(op->getLoc(), quotient, rhs));
    return success();
  }
};

struct AbsDecompose : public OpRewritePattern<AbsOp> {
 public:
  explicit AbsDecompose(MLIRContext *context)
      : OpRewritePattern<AbsOp>(context) {}

  LogicalResult matchAndRewrite(AbsOp op,
                                PatternRewriter &rewriter) const override {
    auto sign = rewriter.create<SignOp>(op->getLoc(), op.getOperand(),
                                        /*ignore_zero*/ true);
    rewriter.replaceOpWithNewOp<MulOp>(op, sign, op.getOperand());
    return success();
  }
};

struct Convolution2DDecompose : public OpRewritePattern<ConvolutionOp> {
 private:
  template <typename Itr>
  inline int64_t product(Itr first, Itr last) const {
    return std::accumulate(first, last, 1, std::multiplies<>());
  }

  // Example:
  // in  = {1, 3}, n = 5
  // res = {1, 3, 0, 2, 4}
  llvm::SmallVector<int64_t> buildFullIndex(llvm::ArrayRef<int64_t> in,
                                            int64_t n) const {
    llvm::SmallVector<int64_t> out(in.begin(), in.end());
    out.reserve(n);
    for (int64_t dim = 0; dim < n; dim++) {
      SPU_ENFORCE_LT(dim, n, "dim={} out of bound={}", dim, n);
      if (std::find(in.begin(), in.end(), dim) == in.end()) {
        out.emplace_back(dim);
      }
    }
    return out;
  }

  // Tensor contraction x and y on index ix and iy.
  // See awesome [tutorial](https://www.tensors.net/tutorial-1) for details.
  Value tensordot(PatternRewriter &rewriter, Value x, Value y,
                  llvm::ArrayRef<int64_t> ix,
                  llvm::ArrayRef<int64_t> iy) const {
    auto x_type = mlir::dyn_cast<RankedTensorType>(x.getType());
    auto y_type = mlir::dyn_cast<RankedTensorType>(y.getType());

    auto perm_x = buildFullIndex(ix, x_type.getRank());
    auto perm_y = buildFullIndex(iy, y_type.getRank());

    // number of dims to contract.
    const size_t nc = ix.size();

    std::rotate(perm_x.begin(), perm_x.begin() + nc, perm_x.end());

    // convert to mmul shape.
    Value xx = rewriter.create<TransposeOp>(x.getLoc(), x, perm_x);
    auto xxt = mlir::dyn_cast<RankedTensorType>(xx.getType());

    xx = rewriter.create<ReshapeOp>(
        x.getLoc(),
        RankedTensorType::get(
            {product(xxt.getShape().begin(), xxt.getShape().end() - nc),
             product(xxt.getShape().end() - nc, xxt.getShape().end())},
            xxt.getElementType()),
        xx);

    Value yy = rewriter.create<TransposeOp>(y.getLoc(), y, perm_y);
    auto yyt = mlir::dyn_cast<RankedTensorType>(yy.getType());

    yy = rewriter.create<ReshapeOp>(
        y.getLoc(),
        RankedTensorType::get(
            {product(yyt.getShape().begin(), yyt.getShape().begin() + nc),
             product(yyt.getShape().begin() + nc, yyt.getShape().end())},
            yyt.getElementType()),
        yy);

    // do matrix multiplication.
    auto zz = rewriter.create<DotOp>(x.getLoc(), xx, yy);

    // decompose shape back.
    llvm::SmallVector<int64_t> res_shape(xxt.getShape().begin(),
                                         xxt.getShape().end() - nc);
    res_shape.insert(res_shape.end(), yyt.getShape().begin() + nc,
                     yyt.getShape().end());

    return rewriter.create<ReshapeOp>(
        zz->getLoc(),
        RankedTensorType::get(
            res_shape,
            mlir::dyn_cast<RankedTensorType>(zz.getType()).getElementType()),
        zz);
  }

 public:
  explicit Convolution2DDecompose(MLIRContext *context)
      : OpRewritePattern<ConvolutionOp>(context) {}

  LogicalResult matchAndRewrite(ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto dnums = op.getDimensionNumbers();
    if (dnums.getInputSpatialDimensions().size() != 2) {
      return failure();
    }

    auto input_type = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto kernel_type = mlir::dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto result_type =
        mlir::dyn_cast<RankedTensorType>(op.getResult().getType());

    // Alias input dimensions.
    auto N = input_type.getShape()[0];
    auto H = input_type.getShape()[1];
    auto W = input_type.getShape()[2];
    auto C = input_type.getShape()[3];

    auto h = kernel_type.getShape()[0];
    auto w = kernel_type.getShape()[1];
    auto O = kernel_type.getShape()[3];
    if (kernel_type.getShape()[2] != C) {
      // input/kernel channel mismatch
      return failure();
    }

    if (result_type.getShape()[0] != N) {
      // result batch mismatch
      return failure();
    }

    auto hh = result_type.getShape()[1];
    auto ww = result_type.getShape()[2];

    if (result_type.getShape()[3] != O) {
      // result filters mismatch
      return failure();
    }

    int64_t sh = 1;
    int64_t sw = 1;

    if (op.getWindowStrides().has_value()) {
      sh = (*op.getWindowStrides())[0];
      sw = (*op.getWindowStrides())[1];
    }

    if (hh != (H - h) / sh + 1 || ww != (W - w) / sw + 1) {
      return failure();
    }

    // expand the image according to the kernel size.
    // assumption:
    // - padding is erased by some compiler pass.
    // - input  : NxHxWxC
    // - kernel : hxwxCxO
    llvm::SmallVector<Value> images;
    auto sliceShape =
        RankedTensorType::get({N, h, w, C}, input_type.getElementType());
    auto windowShape =
        RankedTensorType::get({N, 1, h, w, C}, input_type.getElementType());
    for (int64_t x = 0; x <= H - h; x += sh) {
      for (int64_t y = 0; y <= W - w; y += sw) {
        auto window = rewriter.create<SliceOp>(
            op->getLoc(), sliceShape, op.getLhs(),
            llvm::ArrayRef<int64_t>{0, x, y, 0},
            llvm::ArrayRef<int64_t>{N, x + h, y + w, C},
            llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        images.emplace_back(
            rewriter.create<ReshapeOp>(op->getLoc(), windowShape, window));
      }
    }

    auto stacked = rewriter.create<ConcatenateOp>(op->getLoc(), images, 1);

    Value expanded = rewriter.create<ReshapeOp>(
        op->getLoc(),
        RankedTensorType::get({N, hh, ww, h, w, C},
                              input_type.getElementType()),
        stacked);

    // Contract on h, w, C
    // expanded:  (N, hh, ww, h, w, C)
    // kernel:               (h, w, C, O)
    // result:    (N, hh, ww,          O)
    auto result =
        tensordot(rewriter, expanded, op.getRhs(), {3, 4, 5}, {0, 1, 2});

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct DecomposeSecretLShift : public OpRewritePattern<pphlo::ShiftLeftOp> {
 private:
  TypeTools tool_;

 public:
  explicit DecomposeSecretLShift(MLIRContext *context)
      : OpRewritePattern<pphlo::ShiftLeftOp>(context), tool_(context) {}

  LogicalResult matchAndRewrite(pphlo::ShiftLeftOp op,
                                PatternRewriter &rewriter) const override {
    auto rhs_type = op.getOperand(1).getType();
    auto rhs_vis = tool_.getTypeVisibility(rhs_type);

    if (rhs_vis == pphlo::Visibility::PUBLIC) {
      return failure();
    }

    auto pub_rhs_type = tool_.getExpressedType(rhs_type);

    auto el_type =
        mlir::dyn_cast<IntegerType>(getElementTypeOrSelf(pub_rhs_type));

    APInt c_2(el_type.getWidth(), 2, !el_type.isUnsigned());

    auto two = rewriter.create<arith::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(
                         mlir::dyn_cast<RankedTensorType>(pub_rhs_type), c_2));

    auto pow_of_two =
        rewriter.create<PowOp>(op.getLoc(), rhs_type, two, op.getOperand(1));

    // left shift is multiply
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getOperand(0),
                                       pow_of_two);

    return success();
  }
};

template <typename ShiftT>
struct DecomposeSecretRShift : public OpRewritePattern<ShiftT> {
 private:
  TypeTools tool_;

 public:
  explicit DecomposeSecretRShift(MLIRContext *context)
      : OpRewritePattern<ShiftT>(context), tool_(context) {}

  LogicalResult matchAndRewrite(ShiftT op,
                                PatternRewriter &rewriter) const override {
    auto rhs_type = op.getOperand(1).getType();
    auto rhs_vis = tool_.getTypeVisibility(rhs_type);

    if (rhs_vis == pphlo::Visibility::PUBLIC) {
      return failure();
    }

    auto pub_rhs_type = tool_.getExpressedType(rhs_type);

    auto el_type =
        mlir::dyn_cast<IntegerType>(getElementTypeOrSelf(pub_rhs_type));

    Value result = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(pub_rhs_type));

    // Slow way to doing secret shift
    int64_t nbits = el_type.getWidth();
    for (int64_t b = 0; b < nbits; ++b) {
      APInt bits_to_shift(el_type.getWidth(), b, !el_type.isUnsigned());
      auto bits_to_shift_v = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          DenseElementsAttr::get(mlir::dyn_cast<RankedTensorType>(pub_rhs_type),
                                 bits_to_shift));
      auto shifted = rewriter.create<ShiftT>(
          op.getLoc(), op.getResult().getType(), op.getLhs(), bits_to_shift_v);

      // mask
      auto mask = rewriter.create<pphlo::EqualOp>(op.getLoc(), op.getRhs(),
                                                  bits_to_shift_v);
      // mul
      auto masked = rewriter.create<pphlo::MulOp>(op.getLoc(), mask, shifted);
      // Add to result
      result = rewriter.create<pphlo::AddOp>(op.getLoc(), masked, result);
    }

    rewriter.replaceOp(op, result);

    return success();
  }
};

class SelectDecompose : public OpRewritePattern<pphlo::SelectOp> {
 public:
  explicit SelectDecompose(MLIRContext *context)
      : OpRewritePattern<pphlo::SelectOp>(context) {}

  LogicalResult matchAndRewrite(pphlo::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto a = op.getOnTrue();
    auto b = op.getOnFalse();
    auto pred = op.getPred();

    // b + pred*(a-b)
    // a-b
    Value ret = rewriter.create<pphlo::SubtractOp>(op->getLoc(), a, b);
    // pred*(a-b)
    ret = rewriter.create<pphlo::MulOp>(op->getLoc(), pred, ret);
    // b + pred*(a-b)
    rewriter.replaceOpWithNewOp<pphlo::AddOp>(op, b, ret);

    return success();
  }
};

struct DecomposeFxpFloor : public OpRewritePattern<FloorOp> {
 private:
  TypeTools tools_;

 public:
  explicit DecomposeFxpFloor(MLIRContext *context)
      : OpRewritePattern<pphlo::FloorOp>(context), tools_(context) {}

  LogicalResult matchAndRewrite(pphlo::FloorOp op,
                                PatternRewriter &rewriter) const override {
    if (!tools_.isFixedPointType(op.getType())) {
      return failure();
    }

    ShapedType result_type = mlir::cast<ShapedType>(op.getType());

    Type int_el_type =
        IntegerType::get(rewriter.getContext(), tools_.getFxpWidth(result_type),
                         IntegerType::SignednessSemantics::Signless);

    auto casted_in_type = tools_.replaceBaseType(result_type, int_el_type);

    auto loc = op->getLoc();

    auto fxp_bits = rewriter.create<arith::ConstantOp>(
        loc, SplatElementsAttr::get(
                 result_type.clone(int_el_type),
                 rewriter.getIntegerAttr(int_el_type,
                                         tools_.getFxpBits(result_type))));

    // lshift(arshift(x, fxp_bits), fxp_bits)
    Value ret =
        rewriter.create<BitcastConvertOp>(loc, casted_in_type, op.getOperand());

    ret = rewriter.create<ShiftRightArithmeticOp>(loc, ret, fxp_bits);

    ret = rewriter.create<ShiftLeftOp>(loc, ret, fxp_bits);

    // bitcast to result type
    rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(op, result_type, ret);

    return success();
  }
};

struct GeneralDecomposeOps
    : public GeneralDecomposeOpsBase<GeneralDecomposeOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    populateWithGenerated(*patterns);
    patterns->add<Convolution2DDecompose>(ctx);
  }
};

struct SecretDecomposeOps : public SecretDecomposeOpsBase<SecretDecomposeOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->add<PowerDecompose, IDivDecompose, CeilDecompose, RoundDecompose,
                  AbsDecompose, RemainderDecompose, /*SelectDecompose,*/
                  DecomposeFxpFloor, DecomposeSecretLShift,
                  DecomposeSecretRShift<pphlo::ShiftRightLogicalOp>,
                  DecomposeSecretRShift<pphlo::ShiftRightArithmeticOp>>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGeneralDecomposeOps() {
  return std::make_unique<GeneralDecomposeOps>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createSecretDecomposeOps() {
  return std::make_unique<SecretDecomposeOps>();
}

}  // namespace mlir::spu::pphlo
