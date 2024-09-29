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
#include "mlir/Transforms/DialectConversion.h"

// depending dialects
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {
namespace {

class BroadcastOpConverter : public OpConversionPattern<pphlo::BroadcastOp> {
 private:
  int64_t getBroadcastSizes(pphlo::BroadcastOp op) const {
    return op.getType().getRank() - op.getBroadcastDimensions().size();
  }

  Value collapseExpandingDims(
      PatternRewriter &rewriter, Location loc, Value operand,
      SmallVector<int64_t> &dimensions,
      llvm::function_ref<bool(int64_t)> isExpandingDim) const {
    auto operandTy = mlir::cast<RankedTensorType>(operand.getType());

    SmallVector<ReassociationIndices> reassociationMap;
    ReassociationIndices currentIndices;

    ArrayRef<int64_t> operandShape = operandTy.getShape();
    SmallVector<int64_t> newOperandShape;
    SmallVector<int64_t> newDimensions;

    for (auto [idx, dim] : llvm::enumerate(dimensions)) {
      currentIndices.push_back(idx);

      if (!isExpandingDim(idx)) {
        reassociationMap.push_back(currentIndices);
        currentIndices.clear();
        newOperandShape.push_back(operandShape[idx]);
        newDimensions.push_back(dim);
      }
    }

    if (!reassociationMap.empty()) {
      reassociationMap.back().insert(reassociationMap.back().end(),
                                     currentIndices.begin(),
                                     currentIndices.end());
    }

    if (dimensions.size() != newDimensions.size()) {
      dimensions = newDimensions;

      auto newOperandType =
          RankedTensorType::get(newOperandShape, operandTy.getElementType());
      operand = rewriter.create<tensor::CollapseShapeOp>(
          loc, newOperandType, operand, reassociationMap);
    }
    return operand;
  }

  // Insert linalg.transpose if broadcasted dimensions are not in sorted order.
  // linalg.broadcast does not support implicit transpose, so the input
  // needs to be explicitly transposed.
  Value transposeBroadcastOperand(PatternRewriter &rewriter, Location loc,
                                  Value operand,
                                  SmallVector<int64_t> &dimensions) const {
    // Do not insert `transpose` is dimensions are already sorted.
    if (llvm::is_sorted(dimensions)) {
      return operand;
    }

    SmallVector<int64_t> permutation =
        llvm::to_vector(llvm::seq<int64_t>(0, dimensions.size()));
    llvm::sort(permutation, [&](int64_t lhs, int64_t rhs) {
      return dimensions[lhs] < dimensions[rhs];
    });

    auto operandTy = mlir::cast<ShapedType>(operand.getType());
    ArrayRef<int64_t> operandShape = operandTy.getShape();
    SmallVector<int64_t> transposedOperandShape;
    SmallVector<int64_t> transposedDimensions;

    for (int64_t index : permutation) {
      transposedOperandShape.push_back(operandShape[index]);
      transposedDimensions.push_back(dimensions[index]);
    }
    dimensions = transposedDimensions;

    auto empty = rewriter.create<tensor::EmptyOp>(loc, transposedOperandShape,
                                                  operandTy.getElementType());

    return rewriter
        .create<mlir::linalg::TransposeOp>(
            loc, operand, empty, rewriter.getDenseI64ArrayAttr(permutation))
        ->getResult(0);
  }

  LogicalResult lowerScalarBroadcast(ConversionPatternRewriter &rewriter,
                                     const TypeConverter *typeConverter,
                                     pphlo::BroadcastOp op) const {
    auto resultTy = typeConverter->convertType<ShapedType>(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    SmallVector<int64_t> dimensions = llvm::to_vector(
        llvm::seq<int64_t>(0, op.getResult().getType().getRank()));

    Location loc = op.getLoc();
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType());

    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(op, op.getOperand(),
                                                     emptyTensor, dimensions);
    return success();
  }

 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::BroadcastOp op, pphlo::BroadcastOpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getOperand().getType().getRank() == 0) {
      return lowerScalarBroadcast(rewriter, typeConverter, op);
    }

    Location loc = op.getLoc();

    SmallVector<int64_t> broadcastDimensions =
        llvm::to_vector(op.getBroadcastDimensions());

    Value operand = op.getOperand();
    auto operandTy = mlir::cast<ShapedType>(operand.getType());
    auto resultTy = mlir::cast<ShapedType>(op.getType());

    ArrayRef<int64_t> operandShape = operandTy.getShape();
    ArrayRef<int64_t> resultShape = resultTy.getShape();

    operand = collapseExpandingDims(
        rewriter, loc, operand, broadcastDimensions, [&](int64_t i) {
          return operandShape[i] == 1 &&
                 resultShape[broadcastDimensions[i]] != 1;
        });
    operand =
        transposeBroadcastOperand(rewriter, loc, operand, broadcastDimensions);

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType());

    SmallVector<int64_t> addedDimensions;
    for (int64_t dim : llvm::seq<int64_t>(0, resultTy.getRank())) {
      if (!llvm::is_contained(broadcastDimensions, dim)) {
        addedDimensions.push_back(dim);
      }
    }

    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(op, operand, emptyTensor,
                                                     addedDimensions);
    return success();
  }
};

class TransposeOpConverter : public OpConversionPattern<pphlo::TransposeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::TransposeOp op, pphlo::TransposeOpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = mlir::dyn_cast<ShapedType>(op.getType());

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), result_type.getShape(), result_type.getElementType());

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, op.getOperand(), emptyTensor, op.getPermutationAttr());

    return success();
  }
};

class ReverseOpConverter : public OpConversionPattern<pphlo::ReverseOp> {
  SmallVector<AffineMap, 2> getIndexingMaps(pphlo::ReverseOp op,
                                            Builder *b) const {
    auto resultType = llvm::cast<ShapedType>(op.getType());
    int64_t nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.reserve(nloops);
    for (int64_t i = 0; i < nloops; ++i) {
      inputExprs.push_back(b->getAffineDimExpr(i));
    }
    for (const auto &dim : op.getDimensions()) {
      if (resultType.isDynamicDim(dim)) {
        return {};
      }
      int n = resultType.getShape()[dim];
      inputExprs[dim] = b->getAffineConstantExpr(n - 1) - inputExprs[dim];
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, inputExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }

  SmallVector<utils::IteratorType, 3> getNParallelLoopsAttrs(
      unsigned nLoops) const {
    SmallVector<utils::IteratorType, 3> res(nLoops,
                                            utils::IteratorType::parallel);
    return res;
  }

 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::ReverseOp op, pphlo::ReverseOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ShapedType resultType = op.getType();
    resultType =
        this->getTypeConverter()->template convertType<ShapedType>(resultType);
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    SmallVector<AffineMap, 2> indexingMaps = getIndexingMaps(op, &rewriter);
    if (indexingMaps.empty()) {
      return rewriter.notifyMatchFailure(op, "could not derive indexing maps");
    }

    int64_t nloops = resultType.getRank();
    Location loc = op.getLoc();

    auto empty_tensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/adaptor.getOperands().front(),
        /*outputBuffers=*/

        ValueRange{empty_tensor}, indexingMaps, getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
        });
    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
    return success();
  }
};

struct ReduceOpConverter : public OpConversionPattern<pphlo::ReduceOp> {
  using OpConversionPattern<pphlo::ReduceOp>::OpConversionPattern;

 public:
  LogicalResult matchAndRewrite(
      pphlo::ReduceOp op, pphlo::ReduceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto reductionDims = llvm::to_vector(op.getDimensions());
    // stablehlo.reduce doesn't specify the order of the reduction dimensions.
    llvm::sort(reductionDims);

    auto toRankedTensor = [](Value v) -> RankedTensorType {
      return cast<RankedTensorType>(v.getType());
    };

    SmallVector<Value> outputs;
    SmallVector<RankedTensorType> operandTypes;
    SmallVector<RankedTensorType> initTypes;
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
      return failure();
    }

    Location loc = op.getLoc();
    for (auto [operand, initValue, resultType] : llvm::zip_equal(
             adaptor.getInputs(), adaptor.getInitValues(), resultTypes)) {
      auto initType = toRankedTensor(initValue);
      initTypes.push_back(initType);
      auto operandType = toRankedTensor(operand);
      operandTypes.push_back(operandType);
      initValue = rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);
      auto tensorResultType = cast<RankedTensorType>(resultType);
      // For linalg.reduce, the result type's dimensions must match the input's
      // dimensions, whereas StableHLO allows replacing static dimensions with
      // dynamic ones.
      SmallVector<int64_t> resultShape;
      for (auto [index, dim] :
           llvm::enumerate(cast<ShapedType>(operand.getType()).getShape())) {
        if (!llvm::is_contained(reductionDims, index)) {
          resultShape.push_back(dim);
        }
      }

      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultShape, tensorResultType.getElementType());
      Value filledTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, emptyTensor).result();
      outputs.push_back(filledTensor);
    }

    auto linalgOp = rewriter.create<linalg::ReduceOp>(loc, adaptor.getInputs(),
                                                      outputs, reductionDims,
                                                      /*bodyBuild=*/nullptr);

    Region &region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getBody(), region, region.end());

    // Convert the signature of the body. The reduce op 'computation' region
    // apply function has a signature with tensor types, this is converted to a
    // function with element types. E.g. the signature "(tensor<f32>,
    // tensor<f32>) -> tensor<f32>" will be converted to "(f32, f32) -> f32".
    // Also, we need to swap the operands of the function. The stablehlo.reduce
    // op expects the init values to be the first parameters of the apply
    // function, while the linalg.reduction op expects the init values as the
    // last parameters of the 'combiner' region apply function.
    TypeConverter::SignatureConversion signatureConverter(
        linalgOp.getNumDpsInputs() * 2);
    assert(linalgOp.getNumDpsInputs() == linalgOp.getNumDpsInits());
    for (const auto &[idx, val] : llvm::enumerate(operandTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx + linalgOp.getNumDpsInputs(),
          // type for new operand number 'idx'.
          typeConverter->convertType(val.getElementType()));
    }
    for (const auto &[idx, val] : llvm::enumerate(initTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx,
          // type for new operand number 'idx' + linalgOp.getNumInputs()
          typeConverter->convertType(val.getElementType()));
    }

    if (failed(rewriter.convertRegionTypes(&linalgOp.getCombiner(),
                                           *this->getTypeConverter(),
                                           &signatureConverter))) {
      return op->emitOpError("Failed to convert body region");
    }

    // Cast the result to the correct type.
    SmallVector<Value> results;
    for (auto [result, resultType] :
         llvm::zip(linalgOp.getResults(), resultTypes)) {
      results.push_back(
          rewriter.createOrFold<tensor::CastOp>(loc, resultType, result));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ReduceWindowOpConverter
    : public OpConversionPattern<pphlo::ReduceWindowOp> {
  using OpConversionPattern<pphlo::ReduceWindowOp>::OpConversionPattern;

  SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
      unsigned nLoops, unsigned nReduction) const {
    SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
                                            utils::IteratorType::parallel);
    res.append(nReduction, utils::IteratorType::reduction);
    return res;
  }

 public:
  LogicalResult matchAndRewrite(
      pphlo::ReduceWindowOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op->getContext();
    Location loc = op.getLoc();
    llvm::SmallVector<Value> initValues = adaptor.getInitValues();
    llvm::SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
      return failure();
    }
    auto numOperands = initValues.size();

    llvm::SmallVector<int64_t> windowDimensions(op.getWindowDimensions());

    llvm::SmallVector<int64_t> windowStrides(windowDimensions.size(), 1);
    if (op.getWindowStrides()) {
      windowStrides = llvm::to_vector(*op.getWindowStrides());
    }

    llvm::SmallVector<int64_t> windowDilations(windowDimensions.size(), 1);
    if (op.getWindowDilations()) {
      windowDilations = llvm::to_vector(*op.getWindowDilations());
    }

    auto rank = static_cast<int64_t>(windowDimensions.size());
    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> windowExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    SmallVector<int64_t> filteredWindowDims;

    int windowDim = 0;
    for (int64_t i = 0; i < rank; i++) {
      AffineExpr srcExpr = mlir::getAffineDimExpr(i, ctx);

      if (windowStrides[i] != 1) {
        srcExpr = srcExpr * windowStrides[i];
      }

      if (windowDimensions[i] != 1) {
        filteredWindowDims.push_back(windowDimensions[i]);
        AffineExpr windowExpr = mlir::getAffineDimExpr(rank + windowDim, ctx);
        windowExprs.push_back(windowExpr);

        if (windowDilations[i] != 1) {
          windowExpr = windowExpr * windowDilations[i];
        }

        srcExpr = srcExpr + windowExpr;
        windowDim++;
      }

      srcExprs.push_back(srcExpr);
      dstExprs.push_back(mlir::getAffineDimExpr(i, ctx));
    }

    SmallVector<AffineMap> inferredMaps(3, AffineMap::get(ctx));
    if (rank > 0) {
      inferredMaps =
          AffineMap::inferFromExprList({srcExprs, windowExprs, dstExprs}, ctx);
    }

    SmallVector<AffineMap> indexingMaps;

    indexingMaps.append(numOperands, inferredMaps[0]);
    indexingMaps.append(1, inferredMaps[1]);
    indexingMaps.append(numOperands, inferredMaps[2]);

    // Setup the initial values.
    llvm::SmallVector<Value> broadcastValues;
    for (uint64_t i = 0, s = initValues.size(); i < s; i++) {
      Value initValue = initValues[i];
      auto resultTy = llvm::cast<ShapedType>(resultTypes[i]);
      broadcastValues.push_back(rewriter.create<pphlo::BroadcastOp>(
          loc, resultTy, initValue,
          llvm::to_vector(llvm::iota_range<int64_t>(
              0, resultTy.getShape().size(), false))));
    }

    llvm::SmallVector<Value> inputs = llvm::to_vector(adaptor.getInputs());

    // Add the extra input for the reduction dimension.
    inputs.push_back(rewriter.create<tensor::EmptyOp>(loc, filteredWindowDims,
                                                      rewriter.getF32Type()));

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/resultTypes,
        /*inputs=*/inputs,
        /*outputs=*/broadcastValues, indexingMaps,
        getParallelAndReductionIterators(rank + filteredWindowDims.size(),
                                         filteredWindowDims.size()));

    // Convert the signature of the body. This includes converting scalar
    // tensors to their scalar values and inserting an additional block arg for
    // the window arg.
    Region &region = linalgOp.getRegion();
    rewriter.cloneRegionBefore(op.getBody(), region, region.end());

    TypeConverter::SignatureConversion signatureConverter(
        inputs.size() + op->getNumResults() - 1);

    // ReduceWindow requires that the seed be used as a LHS operand inside the
    // region, and the seed is encoded in linalg in the initial out value, so
    // modify the signature of the block and the value mappings, so the output
    // args will correlate with the LHS and the inputs correlate with the RHS.
    for (auto [i, type] : llvm::enumerate(resultTypes)) {
      auto idx = inputs.size() + i - 1;
      signatureConverter.addInputs(idx,
                                   cast<ShapedType>(type).getElementType());
    }

    signatureConverter.addInputs(
        cast<ShapedType>(inputs.back().getType()).getElementType());

    for (auto [i, input] :
         llvm::enumerate(ArrayRef<Value>(inputs).drop_back())) {
      signatureConverter.addInputs(
          i, cast<ShapedType>(input.getType()).getElementType());
    }

    rewriter.applySignatureConversion(&region.front(), signatureConverter,
                                      getTypeConverter());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

class ReturnOpConverter : public OpConversionPattern<pphlo::ReturnOp> {
 public:
  using OpConversionPattern<pphlo::ReturnOp>::OpConversionPattern;

  static bool isLegalReturnOp(Operation *op) {
    return !isa<linalg::ReduceOp, linalg::GenericOp>(
        op->getParentRegion()->getParentOp());
  }

  LogicalResult matchAndRewrite(
      pphlo::ReturnOp op, pphlo::ReturnOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // linalg reduce need a linalg.yield as terminator
    if (!isLegalReturnOp(op)) {
      SmallVector<Value> operands(adaptor.getOperands());
      for (Value &operand : operands) {
        if (isa<ShapedType>(operand.getType())) {
          Location loc = operand.getLoc();
          operand = rewriter.create<tensor::ExtractOp>(loc, operand);
        }
      }
      rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, operands);
      return success();
    }

    return success();
  }
};

template <typename LinalgOpTy>
bool opMatchesLinalgTarget(DotOp op) {
  ArrayRef<int64_t> lhsShape = op.getLhs().getType().getShape();
  ArrayRef<int64_t> rhsShape = op.getRhs().getType().getShape();
  if (lhsShape.size() == 1 && rhsShape.size() == 1 &&
      lhsShape[0] == rhsShape[0]) {
    return std::is_same<LinalgOpTy, linalg::DotOp>::value;
  }
  if (lhsShape.size() == 2 && rhsShape.size() == 1 &&
      lhsShape[1] == rhsShape[0]) {
    return std::is_same<LinalgOpTy, linalg::MatvecOp>::value;
  }
  if (lhsShape.size() == 1 && rhsShape.size() == 2 &&
      lhsShape[0] == rhsShape[0]) {
    return std::is_same<LinalgOpTy, linalg::VecmatOp>::value;
  }
  if (lhsShape.size() == 2 && rhsShape.size() == 2 &&
      lhsShape[1] == rhsShape[0]) {
    return std::is_same<LinalgOpTy, linalg::MatmulOp>::value;
  }
  return false;
}

template <typename LinalgOpTy>
LogicalResult lowerDotOp(ConversionPatternRewriter &rewriter,
                         const TypeConverter *type_converter, DotOp op,
                         DotOpAdaptor adaptor) {
  if (!opMatchesLinalgTarget<LinalgOpTy>(op)) {
    return failure();
  }

  auto result_type =
      mlir::cast<ShapedType>(type_converter->convertType(op.getType()));

  auto loc = op.getLoc();

  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, result_type.getShape(), result_type.getElementType());

  rewriter.replaceOpWithNewOp<LinalgOpTy>(
      op, TypeRange{result_type},
      ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{emptyTensor});

  return success();
}

bool isLegalDotOp(Operation *op) {
  TypeTools tools(op->getContext());
  // Any public pphlo.dot is illegal
  return !tools.isPublicType(op->getResultTypes()[0]);
}

template <typename LinalgOpTy>
class DotOpConverter : public OpConversionPattern<pphlo::DotOp> {
 private:
 public:
  using OpConversionPattern<pphlo::DotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::DotOp op, pphlo::DotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return lowerDotOp<LinalgOpTy>(rewriter, getTypeConverter(), op, adaptor);
  }
};

class LinalgTypeConverter : public TypeConverter {
  static std::optional<Value> scalarToTensor(OpBuilder &builder, Type type,
                                             ValueRange inputs, Location loc) {
    assert(inputs.size() == 1);
    if (mlir::isa<ShapedType>(inputs.front().getType())) {
      return std::nullopt;
    }
    Value result =
        builder
            .create<tensor::FromElementsOp>(
                loc, RankedTensorType::get({}, inputs.front().getType()),
                inputs.front())
            .getResult();
    return result;
  }

 public:
  explicit LinalgTypeConverter() {
    addConversion([](Type t) { return t; });
    addArgumentMaterialization(scalarToTensor);
  }
};

struct LegalizeToLinalg : public LegalizeToLinalgBase<LegalizeToLinalg> {
 private:
  void populateConversionPattern(TypeConverter &converter,
                                 RewritePatternSet &patterns) {
    patterns.insert<
        BroadcastOpConverter,     //
        TransposeOpConverter,     //
        ReduceOpConverter,        //
        ReduceWindowOpConverter,  //
        ReturnOpConverter,        //
        ReverseOpConverter,       //
        DotOpConverter<linalg::MatmulOp>, DotOpConverter<linalg::MatvecOp>,
        DotOpConverter<linalg::VecmatOp>, DotOpConverter<linalg::DotOp>  //
        >(converter, patterns.getContext());
  }

 public:
  LegalizeToLinalg(const LegalizeToLinalg &) = default;
  LegalizeToLinalg() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    LinalgTypeConverter converter;

    target
        .addLegalDialect<mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                         mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
                         pphlo::PPHloDialect>();

    target
        .addIllegalOp<pphlo::ReduceOp, pphlo::ReduceWindowOp, pphlo::ReverseOp,
                      pphlo::BroadcastOp, pphlo::TransposeOp>();

    target.addDynamicallyLegalOp<pphlo::ReturnOp>(
        [&](Operation *op) { return ReturnOpConverter::isLegalReturnOp(op); });

    target.addDynamicallyLegalOp<pphlo::DotOp>(
        [&](Operation *op) { return isLegalDotOp(op); });

    populateConversionPattern(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToLinalg() {
  return std::make_unique<LegalizeToLinalg>();
}

}  // namespace mlir::spu::pphlo
