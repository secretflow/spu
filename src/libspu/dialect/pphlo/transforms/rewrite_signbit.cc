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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {

namespace {

// Signbit can generate the following pattern
// %0 = pphlo.constant dense<0> : tensor<i32>
// %1 = pphlo.constant dense<31> : tensor<i32>
// %2 = pphlo.bitcast_convert %arg0 : (tensor<f32>) -> tensor<i32>
// %3 = pphlo.shift_right_arithmetic %2, %1 : tensor<i32>
// This is not right after legalize to fxp. Rewrite to following
// %0 = pphlo.constant dense<-1.0> : tensor<f32>
// %1 = pphlo.constant dense<1> : tensor<i32>
// %2 = pphlo.sign %arg0 : (tensor<f32>) -> tensor<f32>
// %3 = pphlo.convert %2 : (tensor<f32>) -> tensor<i32>
// %4 = pphlo.shift_right_arithmetic %3, %1 : (tensor<i32>, tensor<i32>) ->
// tensor<i1>
struct ARShiftRightRewrittern
    : public OpRewritePattern<ShiftRightArithmeticOp> {
 private:
  TypeTools tools_;

  std::optional<int64_t> extractSplatConstantValue(
      DenseIntElementsAttr attr) const {
    if (!attr.isSplat()) {
      return std::nullopt;
    }

    return attr.getSplatValue<APInt>().getSExtValue();
  }

  bool isLegitARShift(ShiftRightArithmeticOp op) const {
    auto lhs_type = mlir::dyn_cast<IntegerType>(tools_.getBaseType(
        mlir::dyn_cast<RankedTensorType>(op.getLhs().getType())));
    auto shifted_bits = op.getRhs().getDefiningOp<ConstantOp>();

    if (!shifted_bits) {
      return false;
    }

    auto shifted_bits_v = extractSplatConstantValue(
        mlir::dyn_cast<DenseIntElementsAttr>(shifted_bits.getValue()));

    if (!shifted_bits_v.has_value()) {
      return false;
    }

    return lhs_type.getWidth() - 1 == *shifted_bits_v;
  }

  Value stripConvertOps(Value v) const {
    if (auto parent = v.getDefiningOp<BitcastConvertOp>()) {
      return stripConvertOps(parent.getOperand());
    }
    return v;
  }

 public:
  explicit ARShiftRightRewrittern(MLIRContext *context)
      : OpRewritePattern(context), tools_(context) {}

  LogicalResult matchAndRewrite(ShiftRightArithmeticOp op,
                                PatternRewriter &rewriter) const override {
    if (!isLegitARShift(op)) {
      return failure();
    }

    auto value_before_shift = stripConvertOps(op.getLhs());

    // rewrite
    // sign
    auto sign = rewriter.create<SignOp>(op->getLoc(), value_before_shift, true);
    // convert
    auto convert = rewriter.create<ConvertOp>(op->getLoc(), op.getType(), sign);
    // sign is -1 for negative and 1 for positive
    // arshift 1 bit, to get -1 and 0
    auto one = rewriter.create<ConstantOp>(
        op->getLoc(), rewriter.getOneAttr(op.getRhs().getType()));
    rewriter.replaceOpWithNewOp<ShiftRightArithmeticOp>(op, op.getType(),
                                                        convert, one);
    return success();
  }
};

struct SignbitPattern : public RewriteSignbitPatternsBase<SignbitPattern> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    GreedyRewriteConfig config;
    config.enableFolding();
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<ARShiftRightRewrittern>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRewriteSignbitPatterns() {
  return std::make_unique<SignbitPattern>();
}

}  // namespace mlir::spu::pphlo
