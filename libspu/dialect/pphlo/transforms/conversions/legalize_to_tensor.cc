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

// Base mlir headers
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// depending dialects
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"
#include "libspu/dialect/utils/lowering_intrinsic.h"

namespace mlir::spu::pphlo {
namespace {

Value buildClamp(OpBuilder &builder, Location loc, Value lower, Value current,
                 Value upper) {
  TypeTools tool(builder.getContext());

  if (tool.isPublicType(current.getType())) {
    // Only current can be secret in this pass
    auto m = builder.create<arith::MaxSIOp>(loc, lower, current);
    return builder.create<arith::MinSIOp>(loc, m, upper);
  } else {
    return builder.create<pphlo::ClampOp>(loc, lower, current, upper);
  }
}

class ReshapeOpConverter : public OpRewritePattern<pphlo::ReshapeOp> {
 public:
  using OpRewritePattern<pphlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto dst_shape = mlir::dyn_cast<ShapedType>(op.getType()).getShape();

    auto const_dst_shape = rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        cast<mlir::TypedAttr>(rewriter.getI64TensorAttr(dst_shape)));

    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, op.getType(), op.getOperand(), const_dst_shape);

    return success();
  }
};

class ConcatenateOpConverter : public OpRewritePattern<pphlo::ConcatenateOp> {
 public:
  using OpRewritePattern<pphlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::ConcatenateOp concatOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(
        concatOp, concatOp.getDimension(), concatOp.getOperands());

    return success();
  }
};

class SliceOpConverter : public OpRewritePattern<pphlo::SliceOp> {
 public:
  using OpRewritePattern<pphlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto argType = mlir::dyn_cast<ShapedType>(sliceOp.getOperand().getType());
    SmallVector<OpFoldResult, 3> offsets;
    SmallVector<OpFoldResult, 3> sizes;
    SmallVector<OpFoldResult, 3> strides;
    for (int i = 0, e = argType.getRank(); i < e; ++i) {
      auto start = sliceOp.getStartIndices()[i];
      auto limit = sliceOp.getLimitIndices()[i];
      auto stride = sliceOp.getStrides()[i];
      offsets.push_back(rewriter.getI64IntegerAttr(start));
      // Say that there are k elements in total, we have condition:
      //   start + (k - 1) * strides <= limit - 1
      // ->
      //   k <= (limit - 1 - start + strides) / strides
      sizes.push_back(
          rewriter.getI64IntegerAttr((limit - 1 - start + stride) / stride));
      strides.push_back(rewriter.getI64IntegerAttr(stride));
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        sliceOp, sliceOp.getOperand(), offsets, sizes, strides);

    return success();
  }
};

class PadOpConverter : public OpRewritePattern<pphlo::PadOp> {
 public:
  using OpRewritePattern<pphlo::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::PadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = op.getType();

    // Negative edge padding is decomposed separately.
    auto isNegative = [](const int64_t &i) { return i < 0; };
    if (llvm::any_of(op.getEdgePaddingLow(), isNegative) ||
        llvm::any_of(op.getEdgePaddingHigh(), isNegative)) {
      return failure();
    }

    Value paddingVal =
        rewriter.createOrFold<tensor::ExtractOp>(loc, op.getPaddingValue());

    auto i64ToFoldResult = [&](const int64_t &i) -> OpFoldResult {
      return rewriter.getIntegerAttr(rewriter.getI64Type(), i);
    };

    // If there is no interior padding lower to tensor.pad directly.
    if (llvm::all_of(op.getInteriorPadding(),
                     [](const int64_t &i) { return i == 0; })) {
      auto padTensorOp = rewriter.create<tensor::PadOp>(
          loc, resultType, op.getOperand(),
          llvm::map_to_vector(op.getEdgePaddingLow(), i64ToFoldResult),
          llvm::map_to_vector(op.getEdgePaddingHigh(), i64ToFoldResult),
          paddingVal);
      rewriter.replaceOp(op, padTensorOp.getResult());
      return success();
    }

    // We have interior padding, which can be lowered to tensor.insert_slice.
    // Start by filling a result-sized tensor with the pad value.
    auto emptyTensor = rewriter
                           .create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                    resultType.getElementType())
                           ->getResult(0);

    auto fill =
        rewriter.create<linalg::FillOp>(loc, paddingVal, emptyTensor).result();

    // Get sizes of the original operand.
    auto operandType = llvm::cast<ShapedType>(op.getOperand().getType());
    auto sizes = llvm::map_to_vector(
        llvm::seq<int64_t>(0, operandType.getRank()),
        [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(operandType.getDimSize(dim));
        });
    // Map interior padding to strides.
    auto strides = llvm::map_to_vector(
        op.getInteriorPadding(), [&](const int64_t &stride) -> OpFoldResult {
          return rewriter.getIntegerAttr(rewriter.getI64Type(), stride + 1);
        });

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, op.getOperand(), fill,
        llvm::map_to_vector(op.getEdgePaddingLow(), i64ToFoldResult), sizes,
        strides);
    return success();
  }
};

struct DynamicSliceOpConverter
    : public OpRewritePattern<pphlo::DynamicSliceOp> {
  using OpRewritePattern<pphlo::DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::DynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    auto argType = op.getOperand().getType();

    auto index_type = tools.getBaseType(op.getStartIndices().front().getType());
    auto tensor_index_type = RankedTensorType::get({}, index_type);

    SmallVector<OpFoldResult, 3> offsets;

    SmallVector<Value> secret_index;
    SmallVector<int64_t> secret_dimension;

    SmallVector<int64_t> public_slice_size =
        llvm::to_vector(op.getSliceSizes());

    for (int64_t i = 0, e = argType.getRank(); i < e; ++i) {
      auto start = op.getStartIndices()[i];
      // clamp start to [0, limit - k]
      auto limit = rewriter.create<arith::ConstantOp>(
          op->getLoc(),
          DenseIntElementsAttr::get(
              tensor_index_type,
              APInt(index_type.getIntOrFloatBitWidth(),
                    argType.getShape()[i] - op.getSliceSizes()[i])));
      auto zero = rewriter.create<arith::ConstantOp>(
          op->getLoc(),
          DenseIntElementsAttr::get(
              tensor_index_type,
              APInt::getZero(index_type.getIntOrFloatBitWidth())));
      auto clamped_start =
          buildClamp(rewriter, op->getLoc(), zero, start, limit);

      if (tools.isSecretType(clamped_start.getType())) {
        secret_index.emplace_back(clamped_start);
        secret_dimension.emplace_back(i);
        offsets.emplace_back(rewriter.getIndexAttr(0));
        public_slice_size[i] = argType.getShape()[i];
      } else {
        auto scalar = rewriter.create<tensor::ExtractOp>(
            op->getLoc(), clamped_start, ValueRange{});

        offsets.emplace_back(rewriter.create<arith::IndexCastOp>(
            op->getLoc(), rewriter.getIndexType(), scalar));
      }
    }

    SmallVector<OpFoldResult, 3> sizes = llvm::map_to_vector(
        public_slice_size, [&](const int64_t &size) -> OpFoldResult {
          return rewriter.getIntegerAttr(rewriter.getI64Type(), size);
        });

    SmallVector<OpFoldResult, 3> strides = llvm::map_to_vector(
        argType.getShape(), [&](const int64_t &r) -> OpFoldResult {
          return rewriter.getIntegerAttr(rewriter.getI64Type(), 1);
        });

    Value new_op = rewriter.create<tensor::ExtractSliceOp>(
        op->getLoc(), op.getOperand(), offsets, sizes, strides);

    // If has secret index
    if (!secret_index.empty()) {
      secret_index.push_back(new_op);
      auto call = rewriter.create<pphlo::CustomCallOp>(
          op->getLoc(), TypeRange{op.getType()}, secret_index, SECRET_INDEX);

      // Set attr
      call->setAttr("indexing_dim", rewriter.getI64ArrayAttr(secret_dimension));

      new_op = call.getResult(0);
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

struct DynamicUpdateSliceConverter
    : OpRewritePattern<pphlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::DynamicUpdateSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Check all indices are public
    TypeTools tools(op->getContext());
    for (auto i : op.getStartIndices()) {
      if (!tools.isPublicType(i.getType())) {
        return failure();
      }
    }

    auto operandType = llvm::cast<RankedTensorType>(op.getOperand().getType());
    auto updateType = llvm::cast<RankedTensorType>(op.getUpdate().getType());

    // We do not have to clamp sizes because the semantic of `update`
    // guarantees that it is always in the bounds. See
    // https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice
    SmallVector<OpFoldResult, 3> sizes;
    for (int64_t size : updateType.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(size));
    }

    auto index_type = tools.getBaseType(op.getStartIndices().front().getType());
    auto tensor_index_type = RankedTensorType::get({}, index_type);

    SmallVector<OpFoldResult, 3> startIndices;
    for (auto [idx, start] : llvm::enumerate(op.getStartIndices())) {
      // By pphlo.DynamicUpdateSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - update.dimension_size[i])`
      auto limit = rewriter.create<arith::ConstantOp>(
          op->getLoc(),
          DenseIntElementsAttr::get(
              tensor_index_type,
              APInt(index_type.getIntOrFloatBitWidth(),
                    operandType.getDimSize(idx) - updateType.getDimSize(idx))));
      auto zero = rewriter.create<arith::ConstantOp>(
          op->getLoc(),
          DenseIntElementsAttr::get(
              tensor_index_type,
              APInt::getZero(index_type.getIntOrFloatBitWidth())));
      auto clamped_start =
          buildClamp(rewriter, op->getLoc(), zero, start, limit);
      auto scalar = rewriter.create<tensor::ExtractOp>(
          op->getLoc(), clamped_start, ValueRange{});
      auto casted = rewriter.create<arith::IndexCastOp>(
          op->getLoc(), rewriter.getIndexType(), scalar);
      startIndices.push_back(casted.getResult());
    }

    int64_t rank = operandType.getRank();
    SmallVector<OpFoldResult, 3> strides(rank, rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, op.getUpdate(), op.getOperand(), startIndices, sizes, strides);
    return success();
  }
};

struct LegalizeToTensor : public LegalizeToTensorBase<LegalizeToTensor> {
 private:
  void populateRewritePatterns(RewritePatternSet &patterns) {
    auto *context = patterns.getContext();

    // ShapeOps
    patterns.insert<ReshapeOpConverter,           //
                    ConcatenateOpConverter,       //
                    SliceOpConverter,             //
                    DynamicSliceOpConverter,      //
                    DynamicUpdateSliceConverter,  //
                    PadOpConverter                //
                    >(context);
  }

 public:
  LegalizeToTensor(const LegalizeToTensor &) = default;
  LegalizeToTensor() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);

    populateRewritePatterns(patterns);

    mlir::GreedyRewriteConfig config;
    // There's no point simplifying more than once.
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToTensor() {
  return std::make_unique<LegalizeToTensor>();
}

}  // namespace mlir::spu::pphlo
