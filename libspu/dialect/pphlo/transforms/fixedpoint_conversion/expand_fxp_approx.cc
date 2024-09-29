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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/approximations.h"
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/builder.h"
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/type_converter.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {
namespace {

#define UNARY_APPROX_IMPL(OpKind, ApproxFcn)                                \
  [[maybe_unused]] Value lowerImpl(OpKind op, ValueRange in,                \
                                   OpBuilder &rewriter,                     \
                                   const fixedpoint::Config &config) {      \
    fixedpoint::builder::FxpBuilder builder(rewriter, op->getLoc(), config, \
                                            in[0].getType());               \
    return fixedpoint::ApproxFcn(builder, in[0]);                           \
  }

UNARY_APPROX_IMPL(ExpOp, exponential_approx)
UNARY_APPROX_IMPL(CosineOp, cosine_approx)
UNARY_APPROX_IMPL(SineOp, sine_approx)
UNARY_APPROX_IMPL(LogOp, log_approx)
UNARY_APPROX_IMPL(LogisticOp, logistic_approx)
UNARY_APPROX_IMPL(RsqrtOp, rsqrt_approx)
UNARY_APPROX_IMPL(SqrtOp, sqrt_approx)
UNARY_APPROX_IMPL(TanhOp, tanh_approx)
UNARY_APPROX_IMPL(ReciprocalOp, reciprocal_approx)

#undef UNARY_APPROX_IMPL

[[maybe_unused]] auto lowerImpl(Expm1Op op, ValueRange in, OpBuilder &rewriter,
                                const fixedpoint::Config &config) {
  fixedpoint::builder::FxpBuilder builder(rewriter, op->getLoc(), config,
                                          in[0].getType());
  // ret = exp(x) - 1
  auto exp_op = fixedpoint::exponential_approx(builder, in[0]);
  auto one = builder.fxp_constant(1.0F);
  return builder.substract(exp_op, one);
}

[[maybe_unused]] auto lowerImpl(Log1pOp op, ValueRange in, OpBuilder &rewriter,
                                const fixedpoint::Config &config) {
  fixedpoint::builder::FxpBuilder builder(rewriter, op->getLoc(), config,
                                          in[0].getType());
  // ret = log(x + 1)
  auto one = builder.fxp_constant(1.0F);
  auto x1p = builder.add(in[0], one);
  return fixedpoint::log_approx(builder, x1p);
}

#define BINARY_APPROX_IMPL(OpKind, ApproxFcn)                               \
  [[maybe_unused]] auto lowerImpl(OpKind op, ValueRange ins,                \
                                  OpBuilder &rewriter,                      \
                                  const fixedpoint::Config &config) {       \
    auto lhs = ins[0];                                                      \
    auto rhs = ins[1];                                                      \
    fixedpoint::builder::FxpBuilder builder(rewriter, op->getLoc(), config, \
                                            rhs.getType());                 \
    return fixedpoint::ApproxFcn(builder, lhs, rhs);                        \
  }

BINARY_APPROX_IMPL(DivOp, div_approx)
BINARY_APPROX_IMPL(PowOp, power_approx)
BINARY_APPROX_IMPL(Atan2Op, atan2_approx)

template <typename OP>
class FxpExpander : public OpRewritePattern<OP> {
 private:
  fixedpoint::Config config_;
  TypeTools tools_;

 public:
  FxpExpander(MLIRContext *context, const fixedpoint::Config &config)
      : OpRewritePattern<OP>(context), config_(config), tools_(context) {}

  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    if (tools_.isPublicType(op.getType())) {
      return emitOptionalError(op->getLoc(),
                               "Should not have public op at this stage");
    }

    Value ret;
    if (op->getNumOperands() == 1) {
      ret = lowerImpl(op, ValueRange{op->getOperands()[0]}, rewriter, config_);
    } else if (op->getNumOperands() == 2) {
      auto ins = llvm::to_vector(op->getOperands());

      for (auto &in : ins) {
        if (mlir::isa<FloatType>(getElementTypeOrSelf(in.getType()))) {
          in = convertFloatToFixed(rewriter, op->getLoc(), in,
                                   tools_.getExpressedType(op.getType()));
        }
      }

      ret = lowerImpl(op, ins, rewriter, config_);
    }

    rewriter.replaceOp(op, ret);

    return success();
  }
};

class ErfConverter : public OpRewritePattern<CustomCallOp> {
 private:
  fixedpoint::Config config_;
  TypeTools tools_;

 public:
  ErfConverter(MLIRContext *context, const fixedpoint::Config &config)
      : OpRewritePattern<CustomCallOp>(context),
        config_(config),
        tools_(context) {}

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != "mhlo.erf") {
      return failure();
    }

    if (tools_.isPublicType(op.getResult(0).getType())) {
      return emitOptionalError(op->getLoc(),
                               "Should not have public op at this stage");
    }

    fixedpoint::builder::FxpBuilder builder(rewriter, op->getLoc(), config_,
                                            op.getOperands()[0].getType());

    auto ret = fixedpoint::erf_approx(builder, op.getOperands()[0]);

    rewriter.replaceOp(op, ret);

    return success();
  }
};

struct ExpandFixedPointApprox
    : public ExpandFixedPointApproximationsBase<ExpandFixedPointApprox> {
 private:
  template <typename OpT>
  static void addPatterns(MLIRContext *context, RewritePatternSet &patterns,
                          const fixedpoint::Config &config) {
    patterns.insert<FxpExpander<OpT>>(context, config);
  }

  template <typename OpT, typename OpT2, typename... OpTs>
  static void addPatterns(MLIRContext *context, RewritePatternSet &patterns,
                          const fixedpoint::Config &config) {
    addPatterns<OpT>(context, patterns, config);
    addPatterns<OpT2, OpTs...>(context, patterns, config);
  }

  void populateFixedpointPattern(RewritePatternSet &patterns) {
    auto *context = patterns.getContext();
    auto config = generateFxpApproxConfig();

    addPatterns<Atan2Op, CosineOp, DivOp, Expm1Op, ExpOp, Log1pOp, LogOp,
                LogisticOp, PowOp, RsqrtOp, ReciprocalOp, SineOp, SqrtOp,
                TanhOp>(context, patterns, config);

    patterns.insert<ErfConverter>(context, config);
  }

  fixedpoint::Config generateFxpApproxConfig() const {
    fixedpoint::Config config;
    config.lower_accuracy_rsqrt = lower_accuracy_rsqrt_.getValue();
    config.div_iter = div_iter_.getValue();
    config.exp_mode = fixedpoint::expModeFromString(exp_mode_.getValue());
    config.exp_iter = exp_iter_.getValue();
    config.log_mode = fixedpoint::logModeFromString(log_mode_.getValue());
    config.log_iter = log_iter_.getValue();
    config.log_order = log_order_.getValue();
    config.sig_mode =
        fixedpoint::sigmoidModeFromString(sigmoid_mode_.getValue());
    config.sin_cos_iter = sin_cos_iter_.getValue();
    return config;
  }

 public:
  ExpandFixedPointApprox(const ExpandFixedPointApprox &) = default;
  ExpandFixedPointApprox() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);

    populateFixedpointPattern(patterns);

    mlir::GreedyRewriteConfig config;
    // There's no point simplifying more than once.
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createExpandFixedPointApprox() {
  return std::make_unique<ExpandFixedPointApprox>();
}

}  // namespace mlir::spu::pphlo
