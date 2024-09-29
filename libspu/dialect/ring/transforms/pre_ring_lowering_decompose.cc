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

#include "absl/numeric/bits.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/ring/transforms/pass_details.h"
#include "libspu/dialect/utils/lowering_intrinsic.h"
#include "libspu/dialect/utils/utils.h"

namespace mlir::spu::ring {

namespace {

inline constexpr int Log2Ceil(uint64_t n) {
  return (n <= 1) ? 0 : (64 - absl::countl_zero(n - 1));
}

Value GetUnencodedFloatingPoint(Value v) {
  auto parent = v.getDefiningOp<pphlo::CustomCallOp>();
  if (parent && parent.getCallTargetName() == ENCODE_TO_FXP) {
    return parent.getOperand(0);
  }
  return nullptr;
}

Value BuildEncodeToFxp(OpBuilder &builder, Location loc, Value v,
                       Type fxp_type) {
  return builder
      .create<pphlo::CustomCallOp>(loc, TypeRange{fxp_type}, ValueRange{v},
                                   ENCODE_TO_FXP)
      .getResult(0);
}

struct OrDecompose : public OpRewritePattern<pphlo::OrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::OrOp op,
                                PatternRewriter &rewriter) const override {
    auto x = op.getLhs();
    auto y = op.getRhs();

    // x and y
    Value ret = rewriter.create<pphlo::AndOp>(op->getLoc(), x, y);
    // Y xor (X and Y)
    ret = rewriter.create<pphlo::XorOp>(op->getLoc(), y, ret);
    // X xor Y xor (X and Y)
    rewriter.replaceOpWithNewOp<pphlo::XorOp>(op, x, ret);

    return success();
  }
};

struct BitParityDecompose : public OpRewritePattern<pphlo::BitParityOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::BitParityOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());

    auto pt_type = mlir::dyn_cast<RankedTensorType>(
        tools.getExpressedType(op.getOperand().getType()));

    auto pt_el_type = mlir::cast<IntegerType>(getElementTypeOrSelf(pt_type));

    auto bits = op.getBits();

    Value ret = op.getOperand();

    if (!absl::has_single_bit(bits)) {
      return failure();
    }

    while (bits > 1) {
      auto shift_bits = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          SplatElementsAttr::get(
              pt_type, rewriter.getIntegerAttr(pt_el_type, bits / 2)));
      ret = rewriter.create<pphlo::XorOp>(
          op.getLoc(), ret,
          rewriter.create<pphlo::ShiftRightLogicalOp>(op->getLoc(), ret,
                                                      shift_bits));
      bits /= 2;
    }

    auto ret_type = op.getType();

    Value one = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getOneAttr(tools.getExpressedType(ret_type)));

    rewriter.replaceOpWithNewOp<pphlo::AndOp>(
        op, rewriter.create<pphlo::ConvertOp>(op->getLoc(), ret_type, ret),
        one);

    return success();
  }
};

struct BitDeintlDecompose : public OpRewritePattern<pphlo::BitDeintlOp> {
 private:
  // APInt uses little-endianess, the first part is low bits and the second part
  // is high bits
  std::array<llvm::SmallVector<uint64_t, 2>, 6> kBitIntlSwapMasks = {{
      {0x2222222222222222, 0x2222222222222222},  // 4bit
      {0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C},  // 8bit
      {0x00F000F000F000F0, 0x00F000F000F000F0},  // 16bit
      {0x0000FF000000FF00, 0x0000FF000000FF00},  // 32bit
      {0x00000000FFFF0000, 0x00000000FFFF0000},  // 64bit
      {0xFFFFFFFF00000000, 0x0000000000000000},  // 128bit
  }};

  std::array<llvm::SmallVector<uint64_t, 2>, 6> kBitIntlKeepMasks = {{
      {0x9999999999999999, 0x9999999999999999},  // 4bit
      {0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3},  // 8bit
      {0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F},  // 16bit
      {0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF},  // 32bit
      {0xFFFF00000000FFFF, 0xFFFF00000000FFFF},  // 64bit
      {0x00000000FFFFFFFF, 0xFFFFFFFF00000000},  // 128bit
  }};

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::BitDeintlOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());

    Value out = op.getOperand();
    auto k = tools.getIntWidth(out.getType());

    auto pt_type = mlir::cast<RankedTensorType>(
        tools.getExpressedType(op.getOperand().getType()));

    auto pt_el_type = mlir::cast<IntegerType>(getElementTypeOrSelf(pt_type));

    for (int64_t idx = 0; idx + 1 < Log2Ceil(k); idx++) {
      // auto keep = _constant(ctx, detail::kBitIntlKeepMasks[idx], in.shape());
      // auto move = _constant(ctx, detail::kBitIntlSwapMasks[idx], in.shape());
      auto keep = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          DenseElementsAttr::get(pt_type, llvm::APInt(pt_el_type.getWidth(),
                                                      kBitIntlKeepMasks[idx])));
      auto move = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          DenseElementsAttr::get(pt_type, llvm::APInt(pt_el_type.getWidth(),
                                                      kBitIntlSwapMasks[idx])));
      auto shift = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          DenseElementsAttr::get(pt_type,
                                 llvm::APInt(pt_el_type.getWidth(), 1 << idx)));
      // out = (out & keep) ^ ((out >> shift) & move) ^ ((out & move) << shift);
      out = rewriter.create<pphlo::XorOp>(
          op->getLoc(),
          rewriter.create<pphlo::XorOp>(
              op->getLoc(),
              rewriter.create<pphlo::AndOp>(op->getLoc(), out, keep),
              rewriter.create<pphlo::AndOp>(
                  op->getLoc(),
                  rewriter.create<pphlo::ShiftRightLogicalOp>(op->getLoc(), out,
                                                              shift),
                  move)),
          rewriter.create<pphlo::ShiftLeftOp>(
              op->getLoc(),
              rewriter.create<pphlo::AndOp>(op->getLoc(), out, move), shift));
    }

    rewriter.replaceOp(op, out);

    return success();
  }
};

struct PopcntDecompose : public OpRewritePattern<pphlo::PopcntOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::PopcntOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());

    auto pt_type = mlir::dyn_cast<RankedTensorType>(
        tools.getExpressedType(op.getOperand().getType()));

    auto pt_el_type =
        mlir::dyn_cast<IntegerType>(getElementTypeOrSelf(pt_type));

    Value xb = op.getOperand();

    Value ret = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getZeroAttr(pt_type));

    uint64_t nbits = tools.getIntWidth(op.getOperand().getType());
    if (op.getBits().has_value()) {
      nbits = *op.getBits();
    }

    Value one = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getOneAttr(pt_type));

    for (size_t idx = 0; idx < nbits; ++idx) {
      auto shift_bits = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          DenseElementsAttr::get(
              pt_type, llvm::APInt(pt_el_type.getWidth(), idx, false)));
      Value x_ = rewriter.create<pphlo::ShiftRightLogicalOp>(op->getLoc(), xb,
                                                             shift_bits);
      x_ = rewriter.create<pphlo::AndOp>(op->getLoc(), x_, one);
      ret = rewriter.create<pphlo::AddOp>(op->getLoc(), x_, ret);
    }

    rewriter.replaceOp(op, ret);

    return success();
  }
};

struct PrefixOrDecompose : public OpRewritePattern<pphlo::PrefixOrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::PrefixOrOp op,
                                PatternRewriter &rewriter) const override {
    // Fill all bits after highest bits to 1.
    //
    // Algorithm, lets consider one bit, in each iteration we fill
    // [msb-2^k, msb) to 1.
    //   x0:  010000000   ; x0
    //   x1:  011000000   ; x0 | (x0>>1)
    //   x2:  011110000   ; x1 | (x1>>2)
    //   x3:  011111111   ; x2 | (x2>>4)

    pphlo::TypeTools tools(op->getContext());

    std::uint64_t bit_width = tools.getIntWidth(op.getOperand().getType());
    auto pt_type = mlir::dyn_cast<RankedTensorType>(
        tools.getExpressedType(op.getOperand().getType()));
    auto pt_el_type =
        mlir::dyn_cast<IntegerType>(getElementTypeOrSelf(pt_type));

    Value b0 = op.getOperand();

    for (int idx = 0; idx < absl::bit_width(bit_width) - 1; ++idx) {
      const uint64_t offset = 1L << idx;

      auto offset_v = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          SplatElementsAttr::get(pt_type,
                                 rewriter.getIntegerAttr(pt_el_type, offset)));

      auto b1 = rewriter.create<pphlo::ShiftRightLogicalOp>(op->getLoc(), b0,
                                                            offset_v);

      b0 = rewriter.create<pphlo::OrOp>(op->getLoc(), b0, b1);
    }

    rewriter.replaceOp(op, b0);

    return success();
  }
};

struct LogicalNotDecompose : public OpRewritePattern<pphlo::NotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::NotOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());

    if (tools.getIntOrFxpWidth(op.getType()) != 1) {
      return failure();  // Not what we care here.
    }

    auto pub_type =
        mlir::cast<ShapedType>(tools.getExpressedType(op.getType()));

    auto k1 = rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        SplatElementsAttr::get(pub_type, rewriter.getBoolAttr(true)));

    rewriter.replaceOpWithNewOp<pphlo::XorOp>(op, op.getOperand(), k1);

    return success();
  }
};

struct SignDecompose : public OpRewritePattern<pphlo::SignOp> {
 private:
  Value buildZero(PatternRewriter &rewriter, const mlir::Location &loc,
                  RankedTensorType type) const {
    auto el_type = type.getElementType();

    if (el_type.isInteger()) {
      return rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getZeroAttr(type));
    } else {
      auto fp_constant = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(type.clone(rewriter.getF32Type())));
      return BuildEncodeToFxp(rewriter, loc, fp_constant, type);
    }
  }

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::SignOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());

    auto loc = op->getLoc();
    auto pt_type =
        mlir::cast<RankedTensorType>(tools.getExpressedType(op.getType()));

    // is_negative = x < 0 ? 1 : 0;
    auto zero = buildZero(rewriter, loc, pt_type);
    // x < 0
    Value is_negative =
        rewriter.create<pphlo::LessOp>(loc, op.getOperand(), zero);

    // sign = 1 - 2 * is_negative
    //      = +1 ,if x >= 0
    //      = -1 ,if x < 0
    // Promote bool to I8
    is_negative = rewriter.create<pphlo::ConvertOp>(
        loc, tools.replaceBaseType(is_negative.getType(), rewriter.getI8Type()),
        is_negative);

    auto one = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getOneAttr(pt_type.clone(rewriter.getI8Type())));

    auto neg_two = rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        SplatElementsAttr::get(mlir::cast<ShapedType>(one.getType()),
                               rewriter.getI8IntegerAttr(-2)));

    // -2 * is_negative
    Value mul = rewriter.create<pphlo::MulOp>(loc, is_negative.getType(),
                                              neg_two, is_negative);

    // 1 - 2 * is_negative
    Value result = rewriter.create<pphlo::AddOp>(loc, one, mul);

    if (!op.getIgnoreZero()) {
      // x * (x!=0)
      auto equal_zero =
          rewriter.create<pphlo::EqualOp>(loc, op.getOperand(), zero);

      auto not_equal_zero = rewriter.create<pphlo::NotOp>(loc, equal_zero);

      result = rewriter.create<pphlo::MulOp>(loc, result.getType(),
                                             not_equal_zero, result);
    }

    rewriter.replaceOpWithNewOp<pphlo::ConvertOp>(op, op.getType(), result);

    return success();
  }
};

class SelectDecompose : public OpRewritePattern<pphlo::SelectOp> {
 private:
  Type promoteToNextSignedType(const pphlo::TypeTools &tools, Type in) const {
    auto el_t = mlir::cast<IntegerType>(tools.getBaseType(in));

    IntegerType new_el_t = el_t;

    switch (el_t.getWidth()) {
      case 8:
        new_el_t = IntegerType::get(in.getContext(), 16);
        break;
      case 16:
        new_el_t = IntegerType::get(in.getContext(), 32);
        break;
      case 32:
        new_el_t = IntegerType::get(in.getContext(), 64);
        break;
      case 64:
        new_el_t = IntegerType::get(in.getContext(), 128);
        break;
    }

    return tools.replaceBaseType(in, new_el_t);
  }

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::SelectOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());

    auto loc = op->getLoc();

    Value a = op.getOnTrue();
    Value b = op.getOnFalse();
    auto pred = op.getPred();

    if (tools.isIntType(op.getType())) {
      a = rewriter.create<pphlo::ConvertOp>(
          loc, promoteToNextSignedType(tools, a.getType()), a);
      b = rewriter.create<pphlo::ConvertOp>(
          loc, promoteToNextSignedType(tools, b.getType()), b);
    }

    // b + pred*(a-b)
    // a-b
    Value ret;

    if (tools.isPublicType(a.getType()) && tools.isPublicType(b.getType())) {
      if (tools.isFixedPointType(a.getType())) {
        auto fp_a = GetUnencodedFloatingPoint(a);
        auto fp_b = GetUnencodedFloatingPoint(b);
        auto s = rewriter.create<arith::SubFOp>(loc, fp_a, fp_b);
        ret = BuildEncodeToFxp(rewriter, loc, s, a.getType());
      } else {
        ret = rewriter.create<arith::SubIOp>(loc, a, b);
      }
    } else {
      ret = rewriter.create<pphlo::SubtractOp>(loc, a, b);
    }
    // pred*(a-b)
    ret = rewriter.create<pphlo::MulOp>(loc, pred, ret);
    // b + pred*(a-b)
    ret = rewriter.create<pphlo::AddOp>(loc, b, ret);

    rewriter.replaceOpWithNewOp<pphlo::ConvertOp>(op, op.getType(), ret);

    return success();
  }
};

class SubtractDecompose : public OpRewritePattern<pphlo::SubtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::SubtractOp op,
                                PatternRewriter &rewriter) const override {
    // a - b -> a + (-b)
    pphlo::TypeTools tools(op->getContext());
    auto loc = op->getLoc();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (tools.isPublicType(rhs.getType())) {
      // It is possible to have public on rhs, so this negate might lower
      // to arith
      if (tools.isIntType(rhs.getType())) {
        auto zero = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(rhs.getType()));
        rhs = rewriter.create<arith::SubIOp>(loc, zero, rhs);
      } else {
        auto fp_v = GetUnencodedFloatingPoint(rhs);
        if (fp_v == nullptr) {
          return emitOptionalError(loc, "Find a public from non encode op");
        }
        auto n = rewriter.create<arith::NegFOp>(loc, fp_v);
        rhs = BuildEncodeToFxp(rewriter, loc, n, rhs.getType());
      }
    } else {
      rhs = rewriter.create<pphlo::NegOp>(loc, rhs);
    }

    rewriter.replaceOpWithNewOp<pphlo::AddOp>(op, lhs, rhs);

    return success();
  }
};

struct SplitConvert : public OpRewritePattern<pphlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());

    Value in = op.getOperand();
    auto in_type = mlir::cast<ShapedType>(in.getType());
    auto out_type = mlir::cast<ShapedType>(op.getType());

    auto in_vis = tools.getTypeVisibility(op.getOperand().getType());
    auto out_vis = tools.getTypeVisibility(op.getType());

    auto in_base = tools.getBaseType(in_type);
    auto out_base = tools.getBaseType(out_type);

    if ((in_vis == pphlo::Visibility::SECRET && in_vis == out_vis) ||
        in_base == out_base) {
      // This is a pure secret domain cast or pure vcast, ignore
      return failure();
    }

    auto loc = op->getLoc();

    if (in_vis == pphlo::Visibility::PUBLIC) {
      // convert from in_base to out_base in public domain and then convert to
      // secret
      if (auto in_int = mlir::dyn_cast<IntegerType>(in_base)) {
        auto in_width = in_int.getWidth();
        // cast to signless first
        in = rewriter.create<tensor::BitcastOp>(
            loc, in_type.clone(rewriter.getIntegerType(in_width)), in);

        if (auto out_int = mlir::dyn_cast<IntegerType>(out_base)) {
          auto out_width = out_int.getWidth();
          // Int to int, either a TruncI or Ext(U/S)I
          if (out_width < in_width) {
            in = rewriter.create<arith::TruncIOp>(
                loc, in_type.clone(rewriter.getIntegerType(out_width)), in);
          } else {
            if (out_int.isUnsigned()) {
              in = rewriter.create<arith::ExtUIOp>(
                  loc, in_type.clone(rewriter.getIntegerType(out_width)), in);
            } else {
              in = rewriter.create<arith::ExtSIOp>(
                  loc, in_type.clone(rewriter.getIntegerType(out_width)), in);
            }
          }

          if (out_int.isUnsigned()) {
            // Cast back to unsigned
            in = rewriter.create<tensor::BitcastOp>(
                loc, in_type.clone(out_base), in);
          }
        }

        if (auto out_fxp = mlir::dyn_cast<pphlo::FixedPointType>(out_base)) {
          if (in_int.isUnsigned()) {
            in = rewriter.create<arith::UIToFPOp>(
                loc, in_type.clone(rewriter.getF64Type()), in);
          } else {
            in = rewriter.create<arith::SIToFPOp>(
                loc, in_type.clone(rewriter.getF64Type()), in);
          }

          // Encode to fxp
          in = BuildEncodeToFxp(rewriter, loc, in, in_type.clone(out_base));
        }
      }

      rewriter.replaceOpWithNewOp<pphlo::ConvertOp>(op, out_type, in);
      return success();
    }
    // We should not have s2p

    return emitOptionalError(loc, "should not have s2p conversion");
  }
};

class BitRevDecompose : public OpRewritePattern<pphlo::BitRevOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::BitRevOp op,
                                PatternRewriter &rewriter) const override {
    pphlo::TypeTools tools(op->getContext());
    auto loc = op->getLoc();

    if (!tools.isPublicType(op.getType())) {
      return failure();
    }

    auto nbits = tools.getIntOrFxpWidth(op.getType());

    // Make it signless
    auto el_type = rewriter.getIntegerType(nbits);
    auto i_type = op.getType().clone(el_type);
    auto in =
        rewriter.create<pphlo::BitcastConvertOp>(loc, i_type, op.getOperand());

    // build mask
    auto mask = APInt::getZero(nbits);
    mask.setBits(op.getStart(), op.getEnd());
    mask.flipAllBits();

    Value mask_v = rewriter.create<arith::ConstantOp>(
        loc, SplatElementsAttr::get(i_type, mask));

    Value ret = rewriter.create<arith::AndIOp>(loc, in, mask_v);

    Value accu =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(i_type));

    // for (size_t idx = start; idx < end; idx++) {
    //     if (in & ((U)1 << idx)) {
    //       accu |= (U)1 << (end - 1 - idx + start);
    //     }
    //   }

    for (uint64_t s_idx = op.getStart(), d_idx = op.getEnd() - 1;
         s_idx < op.getEnd(); ++s_idx, --d_idx) {
      // in & (1 << idx)
      Value flag = rewriter.create<arith::ConstantOp>(
          loc, SplatElementsAttr::get(
                   i_type, rewriter.getIntegerAttr(
                               el_type, APInt::getOneBitSet(nbits, s_idx))));
      Value v = rewriter.create<arith::AndIOp>(loc, flag, in);

      if (d_idx > s_idx) {
        // lshift to d_idx
        auto shift_amt = rewriter.create<arith::ConstantOp>(
            loc, SplatElementsAttr::get(
                     i_type, rewriter.getIntegerAttr(
                                 el_type, APInt(nbits, d_idx - s_idx))));
        v = rewriter.create<arith::ShLIOp>(loc, v, shift_amt);
      } else if (d_idx < s_idx) {
        auto shift_amt = rewriter.create<arith::ConstantOp>(
            loc, SplatElementsAttr::get(
                     i_type, rewriter.getIntegerAttr(
                                 el_type, APInt(nbits, s_idx - d_idx))));
        v = rewriter.create<arith::ShRUIOp>(loc, v, shift_amt);
      }
      accu = rewriter.create<arith::OrIOp>(loc, accu, v);
    }

    // (in & ~mask) | accu;
    ret = rewriter.create<arith::OrIOp>(loc, ret, accu);

    rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(op, op.getType(), ret);

    return success();
  }
};

size_t sort_counter = 0;

class OutlineSortKernel : public OpRewritePattern<pphlo::SortOp> {
 private:
  LogicalResult outlineComparator(RewriterBase &b, pphlo::SortOp sort_op,
                                  func::FuncOp &comparator_fcn,
                                  StringRef comparator_name) const {
    IRRewriter rewriter(b);
    Location loc = sort_op.getLoc();
    // First make region isolated from above
    auto captures = makeRegionIsolatedFromAbove(
        b, sort_op.getComparator(), [](Operation *op) {
          // Clone any constant like node into region.
          return op->hasTrait<OpTrait::ConstantLike>();
        });
    if (!captures.empty()) {
      // If there are still captures, bailout...
      return failure();
    }
    FailureOr<func::FuncOp> outlinedFuncOpOrFailure = outlineSingleBlockRegion(
        rewriter, loc, sort_op.getComparator(), comparator_name);
    if (failed(outlinedFuncOpOrFailure)) {
      return failure();
    }
    comparator_fcn = *outlinedFuncOpOrFailure;
    comparator_fcn.setPrivate();
    return success();
  }

 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::SortOp op,
                                PatternRewriter &rewriter) const override {
    auto strCount = std::to_string(sort_counter++);
    auto kernel_name = std::string("outlined_comparator") + strCount;
    func::FuncOp sort_kernel;
    if (failed(outlineComparator(rewriter, op, sort_kernel, kernel_name))) {
      return failure();
    }

    auto call = rewriter.replaceOpWithNewOp<pphlo::CustomCallOp>(
        op, op->getResultTypes(), op->getOperands(), GENERIC_SORT);
    call->setAttrs(op->getAttrs());
    call->setAttr("comparator", rewriter.getStringAttr(kernel_name));

    return success();
  }
};

struct SimpleSortRewriter : public OpRewritePattern<pphlo::SimpleSortOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::SimpleSortOp op,
                                PatternRewriter &rewriter) const override {
    auto call = rewriter.create<pphlo::CustomCallOp>(
        op->getLoc(), op->getResultTypes(), op.getOperands(), SIMPLE_SORT);

    // Build attr
    auto attr = DictionaryAttr::get(
        op->getContext(),
        {// sorting dim
         NamedAttribute(rewriter.getStringAttr("dim"), op.getDimensionAttr()),
         // is ascending
         NamedAttribute(rewriter.getStringAttr("is_ascending"),
                        rewriter.getBoolAttr(op.getSortDirection() ==
                                             pphlo::SortDirection::ASC)),
         // num keys
         NamedAttribute(rewriter.getStringAttr("num_keys"),
                        op.getNumKeysAttr())});

    call->setAttr("spu.sort.attributes", attr);

    rewriter.replaceOp(op, call);

    return success();
  }
};

class RoundNearestEvenRewriter
    : public OpRewritePattern<pphlo::RoundNearestEvenOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::RoundNearestEvenOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::CustomCallOp>(
        op, op->getResultTypes(), op->getOperands(), ROUND_NE);

    return success();
  }
};

struct DecomposeOps : public PreRingLoweringOpDecomposeBase<DecomposeOps> {
  void runOnOperation() override {
    // Reset counter
    sort_counter = 0;
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->add<OrDecompose, BitParityDecompose, BitDeintlDecompose,
                  PopcntDecompose, PrefixOrDecompose, LogicalNotDecompose,
                  BitRevDecompose, SignDecompose, SelectDecompose,
                  SubtractDecompose, SplitConvert, RoundNearestEvenRewriter,
                  OutlineSortKernel, SimpleSortRewriter>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPreRingLoweringOpDecompose() {
  return std::make_unique<DecomposeOps>();
}

}  // namespace mlir::spu::ring
