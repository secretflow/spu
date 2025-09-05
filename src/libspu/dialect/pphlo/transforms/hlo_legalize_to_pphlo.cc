// Copyright 2021 Ant Group Co., Ltd.
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

// This file implements logic for lowering HLO dialect to pphlo dialect.

#include <cstdint>
#include <string>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "libspu/core/prelude.h"
#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/attrs.h"
#include "libspu/dialect/pphlo/IR/base_enums.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/map_stablehlo_to_pphlo_op.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"
#include "libspu/dialect/pphlo/transforms/value_visibility_map.h"
#include "libspu/dialect/pphlo/transforms/visibility_inference.h"
#include "libspu/dialect/utils/utils.h"

namespace mlir::spu::pphlo {
namespace {

bool isAll(llvm::ArrayRef<int64_t> values, int64_t value) {
  return llvm::all_of(values, [value](int64_t v) { return v == value; });
}

ValueVisibilityMap VisibilityDiscovery(
    const llvm::ArrayRef<std::string> input_vis_list, ModuleOp op) {
  // Get the main function
  auto entry_func = get_entrypoint(op);
  SPU_ENFORCE(entry_func != nullptr, "Cannot find main entry point");

  ValueVisibilityMap vis_map;
  // Populate top level io visibility
  for (const auto &blockargs : entry_func.getBody().getArguments()) {
    SPU_ENFORCE(blockargs.getArgNumber() < input_vis_list.size(),
                "Input visibility list does not match actual inputs.");
    Visibility v = Visibility::PUBLIC;

    // There is no compile time private support at this moment.
    // Force compiler to treat private as secret for now
    if (input_vis_list[blockargs.getArgNumber()] == "VIS_PRIVATE" ||
        input_vis_list[blockargs.getArgNumber()] == "VIS_SECRET") {
      v = Visibility::SECRET;
    }
    vis_map.setValueVisibility(blockargs, v);
    vis_map.appendInputVisibility(v);
  }

  VisibilityInference inference(op->getContext(), vis_map);
  inference.infer(entry_func);

  auto ret =
      llvm::dyn_cast<mlir::func::ReturnOp>(entry_func.getBody().back().back());

  for (const auto &op : ret->getOperands()) {
    vis_map.appendOutputVisibility(vis_map.getValueVisibility(op));
  }

  return vis_map;
}

/// Type converter for mhlo type to pphlo types
class HloToPPHloTypeConverter : public TypeConverter {
 private:
  TypeTools typetools_;

  static std::optional<Value> materializeCastFromIllegal(OpBuilder &builder,
                                                         Type type,
                                                         ValueRange inputs,
                                                         Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
  }

  static std::optional<Value> materializeCastToIllegal(OpBuilder &builder,
                                                       Type type,
                                                       ValueRange inputs,
                                                       Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
  }

 public:
  explicit HloToPPHloTypeConverter(MLIRContext *ctx) : typetools_(ctx) {
    // Keep all types unchanged.
    addConversion([](Type type) -> Type { return type; });

    addArgumentMaterialization(materializeCastFromIllegal);
    addSourceMaterialization(materializeCastToIllegal);
    addTargetMaterialization(materializeCastFromIllegal);
  }
};

class FuncOpConverter : public OpConversionPattern<::mlir::func::FuncOp> {
 private:
  const ValueVisibilityMap &vis_;
  TypeTools tools_;

 public:
  FuncOpConverter(TypeConverter &type_converter, MLIRContext *context,
                  const ValueVisibilityMap &vis)
      : OpConversionPattern<::mlir::func::FuncOp>(type_converter, context),
        vis_(vis),
        tools_(context) {}

  LogicalResult matchAndRewrite(
      ::mlir::func::FuncOp op, ::mlir::func::FuncOpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);

    auto functionType = op.getFunctionType();
    auto &region = op.getBody();

    // Convert non-entry blocks
    for (Block &block :
         llvm::make_early_inc_range(llvm::drop_begin(region, 1))) {
      TypeConverter::SignatureConversion conversion(
          /*numOrigInputs=*/block.getNumArguments());
      for (BlockArgument blockArgument : block.getArguments()) {
        auto idx = blockArgument.getArgNumber();
        auto vis_v = vis_.getValueVisibility(blockArgument);
        auto convertedType = tools_.getType(
            typeConverter->convertType(blockArgument.getType()), vis_v);

        conversion.addInputs(idx, convertedType);
      }

      rewriter.applySignatureConversion(&block, conversion, getTypeConverter());
    }

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(functionType.getNumInputs());
    for (const auto &blockarg : llvm::enumerate(op.getBody().getArguments())) {
      auto vis_v = vis_.getValueVisibility(blockarg.value());
      auto convertedType = tools_.getType(
          typeConverter->convertType(blockarg.value().getType()), vis_v);
      conversion.addInputs(blockarg.index(), convertedType);
    }

    // If the SignatureConversion doesn't apply, bail out.
    if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter(),
                                           &conversion))) {
      rewriter.cancelOpModification(op);
      return failure();
    }

    // Update the signature of the function.
    SmallVector<Type, 2> newResultTypes;
    if (failed(typeConverter->convertTypes(functionType.getResults(),
                                           newResultTypes))) {
      rewriter.cancelOpModification(op);
      return failure();
    }

    // Update return types
    auto retOp =
        llvm::dyn_cast<::mlir::func::ReturnOp>(op.getBody().back().back());
    SPU_ENFORCE(retOp->getNumOperands() == newResultTypes.size());

    for (const auto &resultType : llvm::enumerate(newResultTypes)) {
      auto vis_v =
          vis_.getValueVisibility(retOp.getOperand(resultType.index()));
      newResultTypes[resultType.index()] =
          tools_.getType(resultType.value(), vis_v);
    }
    op.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                        newResultTypes));
    rewriter.finalizeOpModification(op);

    return success();
  }
};

class BasePPHloOpConverter {
 protected:
  const ValueVisibilityMap &vis_;
  TypeTools typetools_;
  const TypeConverter &converter_;

 public:
  BasePPHloOpConverter(MLIRContext *ctx, const ValueVisibilityMap &vis,
                       const TypeConverter &converter)
      : vis_(vis), typetools_(ctx), converter_(converter) {}

  llvm::SmallVector<Value, 2> materializeInputs(
      Operation *op, ValueRange adaptor_range) const {
    OpBuilder builder(op);
    SmallVector<Value, 2> operands(op->getNumOperands());

    // Get override vis if avaible
    auto override_vis = vis_.getOperationInputVisibility(op);

    TypeTools typetools_(op->getContext());
    for (size_t idx = 0; idx < operands.size(); ++idx) {
      Visibility vis;
      if (override_vis.has_value()) {
        vis = override_vis.value()[idx];
      } else {
        vis = vis_.getValueVisibility(op->getOperand(idx));
      }
      operands[idx] = converter_.materializeTargetConversion(
          builder, op->getLoc(),
          typetools_.getType(adaptor_range[idx].getType(), vis),
          adaptor_range[idx]);
    }

    return operands;
  }

  llvm::SmallVector<Type> convertResultType(
      ::mlir::Operation::result_range result_range) const {
    llvm::SmallVector<Type> result_types(result_range.size());

    for (size_t idx = 0; idx < result_types.size(); ++idx) {
      auto result_value = result_range[idx];
      auto result_vis = vis_.getValueVisibility(result_value);
      result_types[idx] = typetools_.getType(
          converter_.convertType(result_value.getType()), result_vis);
    }

    return result_types;
  }

  Type convertResultType(Value result_value) const {
    auto result_vis = vis_.getValueVisibility(result_value);
    return typetools_.getType(converter_.convertType(result_value.getType()),
                              result_vis);
    ;
  }
};

template <typename HloOpTy>
class HloToPPHloOpConverter : public OpConversionPattern<HloOpTy>,
                              BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<HloOpTy>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      HloOpTy hlo_op, typename HloToPPHloOpConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(hlo_op.getResult());
    auto operands = materializeInputs(hlo_op, adaptor.getOperands());

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<HloOpTy>>(
        hlo_op, result_type, operands, hlo_op->getAttrs());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mlir::func::ReturnOp>
    : public OpConversionPattern<mlir::func::ReturnOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mlir::func::ReturnOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      ::mlir::func::ReturnOp op, ::mlir::func::ReturnOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    rewriter.modifyOpInPlace(op, [&]() {
      operation->setOperands(materializeInputs(op, adaptor.getOperands()));
    });
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::CompareOp>
    : public OpConversionPattern<stablehlo::CompareOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::CompareOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::CompareOp hlo_op, stablehlo::CompareOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(hlo_op.getResult());

    auto comp_direction = hlo_op.getComparisonDirection();

    auto operands = materializeInputs(hlo_op, adaptor.getOperands());

    if (comp_direction == stablehlo::ComparisonDirection::EQ) {
      rewriter.replaceOpWithNewOp<pphlo::EqualOp>(hlo_op, result_type,
                                                  operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::NE) {
      rewriter.replaceOpWithNewOp<pphlo::NotEqualOp>(hlo_op, result_type,
                                                     operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::LT) {
      rewriter.replaceOpWithNewOp<pphlo::LessOp>(hlo_op, result_type, operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::LE) {
      rewriter.replaceOpWithNewOp<pphlo::LessEqualOp>(hlo_op, result_type,
                                                      operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::GT) {
      rewriter.replaceOpWithNewOp<pphlo::GreaterOp>(hlo_op, result_type,
                                                    operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::GE) {
      rewriter.replaceOpWithNewOp<pphlo::GreaterEqualOp>(hlo_op, result_type,
                                                         operands);
    } else {
      return failure();
    }
    return success();
  }
};

template <>
struct HloToPPHloOpConverter<stablehlo::ReduceOp>
    : public OpConversionPattern<stablehlo::ReduceOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ReduceOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, stablehlo::ReduceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // We may need to materialize operands
    auto materialized_operands = materializeInputs(op, adaptor.getOperands());
    auto result_types = convertResultType(op.getResults());

    // Convert the region signature.
    auto &entry_block = op.getBody().front();
    TypeConverter::SignatureConversion sig_conversion(
        entry_block.getNumArguments());

    for (const auto &arg : entry_block.getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = typetools_.getType(arg_t, vis_.getValueVisibility(arg));
      sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    mlir::NamedAttribute dimAttr(
        StringAttr::get(op->getContext(), "dimensions"),
        DenseI64ArrayAttr::get(getContext(), op.getDimensions()));

    auto new_op = rewriter.create<pphlo::HloToPPHloOp<stablehlo::ReduceOp>>(
        op->getLoc(), result_types, materialized_operands,
        llvm::SmallVector<mlir::NamedAttribute, 1>{dimAttr});

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().end());

    if (failed(rewriter.convertRegionTypes(
            &new_op.getBody(), *this->getTypeConverter(), &sig_conversion))) {
      return failure();
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
struct HloToPPHloOpConverter<stablehlo::ReduceWindowOp>
    : public OpConversionPattern<stablehlo::ReduceWindowOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ReduceWindowOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::ReduceWindowOp op, stablehlo::ReduceWindowOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // We may need to materialize operands
    auto materialized_operands = materializeInputs(op, adaptor.getOperands());
    auto result_types = convertResultType(op->getResults());

    auto num_results = op->getNumResults();

    // Convert the region signature.
    auto &entry_block = op.getBody().front();
    TypeConverter::SignatureConversion sig_conversion(
        entry_block.getNumArguments());

    for (const auto &arg : entry_block.getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = typetools_.getType(arg_t, vis_.getValueVisibility(arg));
      sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    if (op.getBaseDilations().has_value() || op.getPadding().has_value()) {
      auto rank =
          mlir::dyn_cast<RankedTensorType>(op->getOperandTypes()[0]).getRank();
      llvm::SmallVector<int64_t, 2> interior_padding(rank, 0);
      llvm::SmallVector<int64_t, 2> padding_low(rank, 0);
      llvm::SmallVector<int64_t, 2> padding_high(rank, 0);

      bool has_dilation = op.getBaseDilations().has_value() &&
                          !isAll(*op.getBaseDilations(), 1);

      if (has_dilation) {
        for (int64_t rank_idx = 0; rank_idx < rank; ++rank_idx) {
          interior_padding[rank_idx] = (*op.getBaseDilations())[rank_idx] - 1;
        }
      }

      bool has_padding = op.getPadding().has_value() &&
                         (!op.getPaddingAttr().isSplat() ||
                          op.getPaddingAttr().getSplatValue<int64_t>() != 0);

      if (has_padding) {
        for (int64_t rank_idx = 0; rank_idx < rank; ++rank_idx) {
          padding_low[rank_idx] =
              op.getPaddingAttr().getValues<int64_t>()[2 * rank_idx];
          padding_high[rank_idx] =
              op.getPaddingAttr().getValues<int64_t>()[2 * rank_idx + 1];
        }
      }

      if (has_dilation || has_padding) {
        for (size_t idx = 0; idx < num_results; ++idx) {
          materialized_operands[idx] = rewriter.create<pphlo::PadOp>(
              op->getLoc(), materialized_operands[idx],
              materialized_operands[idx + num_results],
              DenseI64ArrayAttr::get(op->getContext(), padding_low),
              DenseI64ArrayAttr::get(op->getContext(), padding_high),
              DenseI64ArrayAttr::get(op->getContext(), interior_padding));
        }
      }
    }

    llvm::SmallVector<mlir::NamedAttribute> attrs;
    {
      // I64ElementsAttr:$window_dimensions,
      attrs.push_back(
          {StringAttr::get(getContext(), "window_dimensions"),
           DenseI64ArrayAttr::get(getContext(), op.getWindowDimensions())});
      // OptionalAttr<I64ElementsAttr>:$window_strides,
      if (op.getWindowStrides().has_value()) {
        attrs.push_back(
            {StringAttr::get(getContext(), "window_strides"),
             DenseI64ArrayAttr::get(getContext(), *op.getWindowStrides())});
      }
      // OptionalAttr<I64ElementsAttr>:$window_dilations,
      if (op.getWindowDilations().has_value()) {
        attrs.push_back(
            {StringAttr::get(getContext(), "window_dilations"),
             DenseI64ArrayAttr::get(getContext(), *op.getWindowDilations())});
      }
    }

    auto new_op =
        rewriter.create<pphlo::HloToPPHloOp<stablehlo::ReduceWindowOp>>(
            op->getLoc(), result_types, materialized_operands, attrs);

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().end());

    if (failed(rewriter.convertRegionTypes(
            &new_op.getBody(), *this->getTypeConverter(), &sig_conversion))) {
      return failure();
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
struct HloToPPHloOpConverter<stablehlo::IfOp>
    : public OpConversionPattern<stablehlo::IfOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::IfOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::IfOp op, stablehlo::IfOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_types = convertResultType(op->getResults());
    auto operands = materializeInputs(op, adaptor.getOperands());

    auto new_op = rewriter.create<pphlo::IfOp>(op->getLoc(), result_types,
                                               operands, op->getAttrs());

    // Copy over the operations inside true/false region.
    rewriter.inlineRegionBefore(op.getTrueBranch(), new_op.getTrueBranch(),
                                new_op.getTrueBranch().end());
    rewriter.inlineRegionBefore(op.getFalseBranch(), new_op.getFalseBranch(),
                                new_op.getFalseBranch().end());

    if (failed(rewriter.convertRegionTypes(&new_op.getTrueBranch(),
                                           *getTypeConverter(), nullptr))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&new_op.getFalseBranch(),
                                           *getTypeConverter(), nullptr))) {
      return failure();
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
struct HloToPPHloOpConverter<stablehlo::CaseOp>
    : public OpConversionPattern<stablehlo::CaseOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::CaseOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::CaseOp op, stablehlo::CaseOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_types = convertResultType(op->getResults());

    // Create new op
    auto operands = materializeInputs(op, adaptor.getOperands());

    auto new_op =
        rewriter.create<pphlo::CaseOp>(op->getLoc(), result_types, operands,
                                       op->getAttrs(), op.getNumRegions());

    // Copy over the operations inside each region.
    for (const auto &r : llvm::enumerate(op.getBranches())) {
      rewriter.inlineRegionBefore(r.value(), new_op.getBranches()[r.index()],
                                  new_op.getBranches()[r.index()].end());
    }

    // Convert each region type
    for (const auto &r : llvm::enumerate(new_op.getBranches())) {
      if (failed(rewriter.convertRegionTypes(&r.value(), *getTypeConverter(),
                                             nullptr))) {
        return failure();
      }
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
struct HloToPPHloOpConverter<stablehlo::WhileOp>
    : public OpConversionPattern<stablehlo::WhileOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::WhileOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::WhileOp op, stablehlo::WhileOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_types = convertResultType(op->getResults());

    // Convert cond region signature.
    auto &cond_region = op.getCond();
    TypeConverter::SignatureConversion cond_sig_conversion(
        cond_region.getNumArguments());

    for (const auto &arg : cond_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = typetools_.getType(arg_t, vis_.getValueVisibility(arg));
      cond_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Convert body region signature.
    auto &body_region = op.getBody();
    TypeConverter::SignatureConversion body_sig_conversion(
        body_region.getNumArguments());

    for (const auto &arg : body_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = typetools_.getType(arg_t, vis_.getValueVisibility(arg));
      body_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // May need to materialize operands
    auto operands = materializeInputs(op, adaptor.getOperands());

    auto new_op = rewriter.create<pphlo::WhileOp>(op->getLoc(), result_types,
                                                  operands, op->getAttrs());

    // Copy over the operations inside body region.
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().end());
    rewriter.inlineRegionBefore(op.getCond(), new_op.getCond(),
                                new_op.getCond().end());

    if (failed(rewriter.convertRegionTypes(
            &new_op.getBody(), *getTypeConverter(), &body_sig_conversion))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(
            &new_op.getCond(), *getTypeConverter(), &cond_sig_conversion))) {
      return failure();
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::BroadcastInDimOp>
    : public OpConversionPattern<stablehlo::BroadcastInDimOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::BroadcastInDimOp>(type_converter,
                                                         context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::BroadcastInDimOp hlo_op,
      stablehlo::BroadcastInDimOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = convertResultType(hlo_op.getResult());

    mlir::NamedAttribute dim(
        StringAttr::get(hlo_op.getContext(), "broadcast_dimensions"),
        DenseI64ArrayAttr::get(getContext(), hlo_op.getBroadcastDimensions()));

    rewriter
        .replaceOpWithNewOp<pphlo::HloToPPHloOp<stablehlo::BroadcastInDimOp>>(
            hlo_op, resultType,
            materializeInputs(hlo_op, adaptor.getOperands()), dim);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::ConstantOp>
    : public OpConversionPattern<stablehlo::ConstantOp> {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<stablehlo::ConstantOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      stablehlo::ConstantOp hlo_op, stablehlo::ConstantOpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<stablehlo::ConstantOp>>(
        hlo_op, hlo_op.getValue());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::IotaOp>
    : public OpConversionPattern<stablehlo::IotaOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::IotaOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::IotaOp hlo_op, stablehlo::IotaOpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(hlo_op.getResult());
    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<stablehlo::IotaOp>>(
        hlo_op, result_type, hlo_op.getIotaDimension());
    return success();
  }
};

/// Need a special conversion rule for Dot to drop precision configs
template <>
class HloToPPHloOpConverter<stablehlo::DotOp>
    : public OpConversionPattern<stablehlo::DotOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::DotOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::DotOp hlo_op, stablehlo::DotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(hlo_op.getResult());

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<stablehlo::DotOp>>(
        hlo_op, result_type, materializeInputs(hlo_op, adaptor.getOperands()));
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::DotGeneralOp>
    : public OpConversionPattern<stablehlo::DotGeneralOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::DotGeneralOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  Value ensureAtLeast3D(ConversionPatternRewriter &rewriter,
                        Value operand) const {
    auto type = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (type.getRank() >= 3) {
      return operand;
    }

    std::vector<int64_t> new_shape(type.getShape());
    new_shape.emplace_back(1);  // Add a trailing one dimension
    auto new_type = RankedTensorType::get(new_shape, type.getElementType());
    return rewriter.create<pphlo::ReshapeOp>(operand.getLoc(), new_type,
                                             operand);
  }

  LogicalResult matchAndRewrite(
      stablehlo::DotGeneralOp hlo_op, stablehlo::DotGeneralOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(hlo_op.getResult());
    auto operands = materializeInputs(hlo_op, adaptor.getOperands());

    auto attr = DotDimensionNumbersAttr::get(
        hlo_op->getContext(),
        hlo_op.getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
        hlo_op.getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
        hlo_op.getDotDimensionNumbersAttr().getLhsContractingDimensions(),
        hlo_op.getDotDimensionNumbersAttr().getRhsContractingDimensions());

    rewriter.replaceOpWithNewOp<pphlo::DotGeneralOp>(
        hlo_op, result_type, ensureAtLeast3D(rewriter, operands[0]),
        ensureAtLeast3D(rewriter, operands[1]), attr);
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::ReturnOp>
    : public OpConversionPattern<stablehlo::ReturnOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ReturnOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::ReturnOp op, stablehlo::ReturnOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto operands = materializeInputs(op, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<pphlo::ReturnOp>(op, std::nullopt, operands);
    return success();
  }
};

template <>
struct HloToPPHloOpConverter<stablehlo::SelectAndScatterOp>
    : public OpConversionPattern<stablehlo::SelectAndScatterOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::SelectAndScatterOp>(type_converter,
                                                           context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::SelectAndScatterOp op,
      stablehlo::SelectAndScatterOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Select
    auto operands = materializeInputs(op, adaptor.getOperands());

    auto result_type = convertResultType(op.getResult());

    bool has_padding = op.getPadding().has_value() &&
                       (!op.getPaddingAttr().isSplat() ||
                        op.getPaddingAttr().getSplatValue<int64_t>() != 0);

    auto converBody = [&](SelectAndScatterOp new_op) {
      // Convert the region signature.
      TypeConverter::SignatureConversion select_sig_conversion(
          op.getSelect().front().getNumArguments());

      for (const auto &arg : op.getSelect().front().getArguments()) {
        auto arg_t = this->getTypeConverter()->convertType(arg.getType());
        auto lower_t = typetools_.getType(arg_t, vis_.getValueVisibility(arg));
        select_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
      }

      TypeConverter::SignatureConversion scatter_sig_conversion(
          op.getScatter().front().getNumArguments());

      for (const auto &arg : op.getScatter().front().getArguments()) {
        auto arg_t = this->getTypeConverter()->convertType(arg.getType());
        auto lower_t = typetools_.getType(arg_t, vis_.getValueVisibility(arg));
        scatter_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
      }

      // Copy over the operations inside the region.
      rewriter.inlineRegionBefore(op.getSelect(), new_op.getSelect(),
                                  new_op.getSelect().end());
      rewriter.inlineRegionBefore(op.getScatter(), new_op.getScatter(),
                                  new_op.getScatter().end());

      if (failed(rewriter.convertRegionTypes(&new_op.getSelect(),
                                             *this->getTypeConverter(),
                                             &select_sig_conversion))) {
        return failure();
      }

      if (failed(rewriter.convertRegionTypes(&new_op.getScatter(),
                                             *this->getTypeConverter(),
                                             &scatter_sig_conversion))) {
        return failure();
      }
      return success();
    };

    if (has_padding) {
      auto rank =
          mlir::dyn_cast<RankedTensorType>(op->getOperandTypes()[0]).getRank();
      llvm::SmallVector<int64_t, 2> padding_low(rank, 0);
      llvm::SmallVector<int64_t, 2> padding_high(rank, 0);
      llvm::SmallVector<int64_t, 2> padding_interior(rank, 0);
      for (int64_t rank_idx = 0; rank_idx < rank; ++rank_idx) {
        padding_low[rank_idx] =
            op.getPaddingAttr().getValues<int64_t>()[2 * rank_idx];
        padding_high[rank_idx] =
            op.getPaddingAttr().getValues<int64_t>()[2 * rank_idx + 1];
      }

      auto operand = rewriter.create<pphlo::PadOp>(
          op->getLoc(), operands[0], operands[2],
          DenseI64ArrayAttr::get(op->getContext(), padding_low),
          DenseI64ArrayAttr::get(op->getContext(), padding_high),
          DenseI64ArrayAttr::get(op->getContext(), padding_interior));

      auto new_op = rewriter.create<pphlo::SelectAndScatterOp>(
          op->getLoc(), operand.getType(), operand, operands[1], operands[2],
          DenseI64ArrayAttr::get(getContext(), *op.getWindowDimensions()),
          DenseI64ArrayAttr::get(getContext(), *op.getWindowStrides()));

      if (failed(converBody(new_op))) {
        return failure();
      }

      llvm::SmallVector<int64_t, 2> slice_end(
          mlir::dyn_cast<RankedTensorType>(new_op.getType()).getShape().begin(),
          mlir::dyn_cast<RankedTensorType>(new_op.getType()).getShape().end());

      for (size_t idx = 0; idx < slice_end.size(); ++idx) {
        slice_end[idx] -= padding_high[idx];
      }

      // Slice back
      rewriter.replaceOpWithNewOp<pphlo::SliceOp>(
          op, result_type, new_op,
          DenseI64ArrayAttr::get(getContext(), padding_low),
          DenseI64ArrayAttr::get(getContext(), slice_end),
          DenseI64ArrayAttr::get(
              getContext(), llvm::SmallVector<int64_t>(slice_end.size(), 1)));
    } else {
      auto new_op = rewriter.create<pphlo::SelectAndScatterOp>(
          op->getLoc(), result_type, operands[0], operands[1], operands[2],
          DenseI64ArrayAttr::get(getContext(), *op.getWindowDimensions()),
          DenseI64ArrayAttr::get(getContext(), *op.getWindowStrides()));

      if (failed(converBody(new_op))) {
        return failure();
      }

      rewriter.replaceOp(op, new_op);
    }

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::RngOp>
    : public OpConversionPattern<stablehlo::RngOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::RngOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::RngOp op, stablehlo::RngOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type resultType = convertResultType(op.getResult());

    rewriter.replaceOpWithNewOp<pphlo::RngOp>(
        op, resultType, materializeInputs(op, adaptor.getOperands()));
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::SortOp>
    : public OpConversionPattern<stablehlo::SortOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::SortOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::SortOp op, stablehlo::SortOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto comp_ret = llvm::dyn_cast<stablehlo::ReturnOp>(
        op.getComparator().back().getTerminator());
    SPU_ENFORCE(comp_ret.getNumOperands() == 1,
                "SortOp comparator can only return one value");

    auto result_types = convertResultType(op->getResults());

    // Convert the region signature.
    auto &comp_region = op.getComparator();
    TypeConverter::SignatureConversion sig_conversion(
        comp_region.getNumArguments());

    for (const auto &arg : comp_region.getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = typetools_.getType(arg_t, vis_.getValueVisibility(arg));
      sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // materialize inputs
    auto operands = materializeInputs(op, adaptor.getOperands());

    auto new_op =
        rewriter.create<pphlo::SortOp>(op->getLoc(), result_types, operands,
                                       op.getDimension(), op.getIsStable());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.getComparator(), new_op.getComparator(),
                                new_op.getComparator().end());

    if (failed(rewriter.convertRegionTypes(&new_op.getComparator(),
                                           *this->getTypeConverter(),
                                           &sig_conversion))) {
      return failure();
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::ConvolutionOp>
    : public OpConversionPattern<stablehlo::ConvolutionOp>,
      BasePPHloOpConverter {
 private:
  /// Returns true if the given `attr` is a splat of the given `value`.
  static bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
    return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
  }

  // Apply dilation and padding to the input of a convolution.
  Value applyConvolutionPadding(
      Location loc, Value input,
      const std::optional<llvm::ArrayRef<int64_t>> &padding,
      const std::optional<llvm::ArrayRef<int64_t>> &lhs_dilation,
      llvm::ArrayRef<int64_t> dim_mappings, OpBuilder &rewriter) const {
    if ((!padding || isAll(*padding, 0)) &&
        (!lhs_dilation || isAll(*lhs_dilation, 1))) {
      return input;
    }

    auto inputType = mlir::dyn_cast<ShapedType>(input.getType());
    size_t rank = inputType.getRank();

    // Translate window padding into low/high padding.
    SmallVector<int64_t, 8> padLow(rank, 0);
    SmallVector<int64_t, 8> padHigh(rank, 0);
    if (padding) {
      // The padding attribute contains two values per dimension, but
      // excludes
      // the batch and feature dimensions.
      assert(
          rank * 2 == padding->size() + 4 &&
          "There should be 2 padding values per dimension, i.e low and high.");
      for (auto i : llvm::seq<int64_t>(0, (*padding).size() / 2)) {
        auto dim = dim_mappings[i];
        padLow[dim] = (*padding)[i * 2];
        padHigh[dim] = (*padding)[i * 2 + 1];
      }
    }

    // Translate input dilation into interior padding.
    SmallVector<int64_t, 8> padInterior(rank, 0);
    if (lhs_dilation) {
      assert(rank == (*lhs_dilation).size() + 2);
      for (auto i : llvm::seq<int64_t>(0, (*lhs_dilation).size())) {
        auto dim = dim_mappings[i];
        padInterior[dim] = (*lhs_dilation)[i] - 1;
      }
    }

    TypeTools type_tools(rewriter.getContext());
    auto zero_attr = rewriter.getZeroAttr(RankedTensorType::get(
        {}, type_tools.getExpressedType(inputType.getElementType())));
    SPU_ENFORCE(zero_attr);
    Value zero = rewriter.create<pphlo::ConstantOp>(loc, zero_attr);
    zero = rewriter.create<pphlo::ConvertOp>(
        loc, RankedTensorType::get({}, inputType.getElementType()), zero);
    return rewriter.create<pphlo::PadOp>(
        loc, input, zero, DenseI64ArrayAttr::get(loc.getContext(), padLow),
        DenseI64ArrayAttr::get(loc.getContext(), padHigh),
        DenseI64ArrayAttr::get(loc.getContext(), padInterior));
  }

 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ConvolutionOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::ConvolutionOp op, stablehlo::ConvolutionOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto old_attr = op.getDimensionNumbers();
    auto attr = ConvDimensionNumbersAttr::get(
        op->getContext(), old_attr.getInputBatchDimension(),
        old_attr.getInputFeatureDimension(),
        old_attr.getInputSpatialDimensions(),
        old_attr.getKernelInputFeatureDimension(),
        old_attr.getKernelOutputFeatureDimension(),
        old_attr.getKernelSpatialDimensions(),
        old_attr.getOutputBatchDimension(),
        old_attr.getOutputFeatureDimension(),
        old_attr.getOutputSpatialDimensions());

    SPU_ENFORCE(op.getFeatureGroupCount() == 1 && op.getBatchGroupCount() == 1);

    Type result_type = convertResultType(op.getResult());

    auto operands = materializeInputs(op, adaptor.getOperands());

    std::optional<llvm::SmallVector<int64_t>> padding{std::nullopt};
    if (adaptor.getPadding().has_value()) {
      padding = llvm::SmallVector<int64_t>{
          adaptor.getPadding()->getValues<int64_t>()};
    }

    auto lhs = applyConvolutionPadding(
        op->getLoc(), operands[0], padding, adaptor.getLhsDilation(),
        adaptor.getDimensionNumbers().getInputSpatialDimensions(), rewriter);

    auto rhs = applyConvolutionPadding(
        op.getLoc(), operands[1], std::nullopt, adaptor.getRhsDilation(),
        op.getDimensionNumbers().getKernelSpatialDimensions(), rewriter);

    auto reversals = op.getWindowReversal();
    if (reversals.has_value()) {
      llvm::SmallVector<int64_t> reversedDims;
      for (const auto &idxAndBool : llvm::enumerate(*reversals)) {
        if (idxAndBool.value()) {
          reversedDims.push_back(
              op.getDimensionNumbers()
                  .getKernelSpatialDimensions()[idxAndBool.index()]);
        }
      }

      rhs = rewriter.create<pphlo::ReverseOp>(
          op.getLoc(), rhs,
          DenseI64ArrayAttr::get(op->getContext(), reversedDims));
    }

    if (op.getWindowStrides().has_value()) {
      rewriter.replaceOpWithNewOp<pphlo::ConvolutionOp>(
          op, result_type, lhs, rhs,
          DenseI64ArrayAttr::get(getContext(), *op.getWindowStrides()), attr);
    } else {
      rewriter.replaceOpWithNewOp<pphlo::ConvolutionOp>(op, result_type, lhs,
                                                        rhs, nullptr, attr);
    }

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::PadOp>
    : public OpConversionPattern<stablehlo::PadOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::PadOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::PadOp op, stablehlo::PadOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type result_type = convertResultType(op.getResult());

    rewriter.replaceOpWithNewOp<pphlo::PadOp>(
        op, result_type, materializeInputs(op, adaptor.getOperands()),
        op->getAttrs());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::BitcastConvertOp>
    : public OpConversionPattern<stablehlo::BitcastConvertOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::BitcastConvertOp>(type_converter,
                                                         context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::BitcastConvertOp op,
      stablehlo::BitcastConvertOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto in_type_size =
        mlir::dyn_cast<RankedTensorType>(op->getOperandTypes()[0])
            .getElementTypeBitWidth();
    auto out_type_size =
        mlir::dyn_cast<RankedTensorType>(op->getResultTypes()[0])
            .getElementTypeBitWidth();

    SPU_ENFORCE(in_type_size == out_type_size,
                "BitcastConvert with different input/output element size is "
                "not supported");

    rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(
        op, convertResultType(op.getResult()),
        materializeInputs(op, adaptor.getOperands()));

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::ConcatenateOp>
    : public OpConversionPattern<stablehlo::ConcatenateOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ConcatenateOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::ConcatenateOp op, stablehlo::ConcatenateOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(op.getResult());

    rewriter.replaceOpWithNewOp<pphlo::ConcatenateOp>(
        op, result_type, materializeInputs(op, adaptor.getOperands()),
        op.getDimension());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::IsFiniteOp>
    : public OpConversionPattern<stablehlo::IsFiniteOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::IsFiniteOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::IsFiniteOp op, stablehlo::IsFiniteOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(op.getResult());
    auto materialized_operands = materializeInputs(op, adaptor.getOperands());

    auto call = rewriter.create<pphlo::CustomCallOp>(
        op.getLoc(), result_type, materialized_operands, IS_FINITE);

    rewriter.replaceOp(op, call);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::GatherOp>
    : public OpConversionPattern<stablehlo::GatherOp>, BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::GatherOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::GatherOp op, stablehlo::GatherOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto old_attr = op.getDimensionNumbers();
    auto result_vis = vis_.getValueVisibility(op.getResult());
    Type resultType = typetools_.getType(
        this->getTypeConverter()->convertType(op.getType()), result_vis);
    auto materialized_operands = materializeInputs(op, adaptor.getOperands());

    auto call = rewriter.create<pphlo::CustomCallOp>(
        op->getLoc(), resultType, materialized_operands, GATHER);
    auto attr = DictionaryAttr::get(
        op->getContext(),
        {NamedAttribute(
             rewriter.getStringAttr("slice_sizes"),
             DenseI64ArrayAttr::get(op->getContext(), op.getSliceSizes())),
         NamedAttribute(rewriter.getStringAttr("offset_dims"),
                        DenseI64ArrayAttr::get(op->getContext(),
                                               old_attr.getOffsetDims())),
         NamedAttribute(
             rewriter.getStringAttr("collapsed_slice_dims"),
             DenseI64ArrayAttr::get(op->getContext(),
                                    old_attr.getCollapsedSliceDims())),
         NamedAttribute(
             rewriter.getStringAttr("index_vector_dim"),
             rewriter.getI64IntegerAttr(old_attr.getIndexVectorDim())),
         NamedAttribute(rewriter.getStringAttr("start_index_map"),
                        DenseI64ArrayAttr::get(op->getContext(),
                                               old_attr.getStartIndexMap()))});
    call->setAttr("pphlo.attributes", attr);
    rewriter.replaceOp(op, call);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::DynamicUpdateSliceOp>
    : public OpConversionPattern<stablehlo::DynamicUpdateSliceOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::DynamicUpdateSliceOp>(type_converter,
                                                             context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::DynamicUpdateSliceOp op,
      stablehlo::DynamicUpdateSliceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type result_type = typetools_.getType(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    auto materialized = materializeInputs(op, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<pphlo::DynamicUpdateSliceOp>(
        op, TypeRange{result_type}, materialized);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::CustomCallOp>
    : public OpConversionPattern<stablehlo::CustomCallOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::CustomCallOp>(type_converter, context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::CustomCallOp op, stablehlo::CustomCallOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> result_types = convertResultType(op->getResults());

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::CustomCallOp>(
        op, result_types, materializeInputs(op, adaptor.getOperands()),
        op.getCallTargetName(), op.getHasSideEffect());

    auto attr = op->getAttr("mhlo.attributes");
    if (attr) {
      new_op->setAttr("mhlo.attributes", attr);
    }

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::PopulationCountOp>
    : public OpConversionPattern<stablehlo::PopulationCountOp>,
      BasePPHloOpConverter {
 public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::PopulationCountOp>(type_converter,
                                                          context),
        BasePPHloOpConverter(context, vis, type_converter) {}

  LogicalResult matchAndRewrite(
      stablehlo::PopulationCountOp hlo_op,
      stablehlo::PopulationCountOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = convertResultType(hlo_op.getResult());
    auto operands = materializeInputs(hlo_op, adaptor.getOperands());

    rewriter.replaceOpWithNewOp<pphlo::PopcntOp>(hlo_op, result_type, operands,
                                                 hlo_op->getAttrs());

    return success();
  }
};

struct HloLegalizeToPPHlo
    : public HloLegalizeToPPHloPassBase<HloLegalizeToPPHlo> {
 private:
  static void populateHLOToPPHloConversionPattern(
      HloToPPHloTypeConverter &converter, RewritePatternSet &patterns,
      const ValueVisibilityMap &vis_map) {
    auto *context = patterns.getContext();

    patterns.insert<FuncOpConverter, HloToPPHloOpConverter<func::ReturnOp>,
                    HloToPPHloOpConverter<stablehlo::AbsOp>,
                    HloToPPHloOpConverter<stablehlo::AddOp>,
                    HloToPPHloOpConverter<stablehlo::AndOp>,
                    HloToPPHloOpConverter<stablehlo::Atan2Op>,
                    HloToPPHloOpConverter<stablehlo::BitcastConvertOp>,
                    HloToPPHloOpConverter<stablehlo::BroadcastInDimOp>,
                    HloToPPHloOpConverter<stablehlo::CaseOp>,
                    HloToPPHloOpConverter<stablehlo::CeilOp>,
                    HloToPPHloOpConverter<stablehlo::ClampOp>,
                    HloToPPHloOpConverter<stablehlo::CompareOp>,
                    HloToPPHloOpConverter<stablehlo::ComplexOp>,
                    HloToPPHloOpConverter<stablehlo::ConcatenateOp>,
                    HloToPPHloOpConverter<stablehlo::ConstantOp>,
                    HloToPPHloOpConverter<stablehlo::ConvertOp>,
                    HloToPPHloOpConverter<stablehlo::ConvolutionOp>,
                    HloToPPHloOpConverter<stablehlo::CosineOp>,
                    HloToPPHloOpConverter<stablehlo::CustomCallOp>,
                    HloToPPHloOpConverter<stablehlo::DivOp>,
                    HloToPPHloOpConverter<stablehlo::DotOp>,
                    HloToPPHloOpConverter<stablehlo::DotGeneralOp>,
                    HloToPPHloOpConverter<stablehlo::DynamicSliceOp>,
                    HloToPPHloOpConverter<stablehlo::DynamicUpdateSliceOp>,
                    HloToPPHloOpConverter<stablehlo::ExpOp>,
                    HloToPPHloOpConverter<stablehlo::Expm1Op>,
                    HloToPPHloOpConverter<stablehlo::FloorOp>,
                    HloToPPHloOpConverter<stablehlo::GatherOp>,
                    HloToPPHloOpConverter<stablehlo::IsFiniteOp>,
                    HloToPPHloOpConverter<stablehlo::IfOp>,
                    HloToPPHloOpConverter<stablehlo::ImagOp>,
                    HloToPPHloOpConverter<stablehlo::IotaOp>,
                    HloToPPHloOpConverter<stablehlo::LogOp>,
                    HloToPPHloOpConverter<stablehlo::Log1pOp>,
                    HloToPPHloOpConverter<stablehlo::LogisticOp>,
                    HloToPPHloOpConverter<stablehlo::MaxOp>,
                    HloToPPHloOpConverter<stablehlo::MinOp>,
                    HloToPPHloOpConverter<stablehlo::MulOp>,
                    HloToPPHloOpConverter<stablehlo::NegOp>,
                    HloToPPHloOpConverter<stablehlo::NotOp>,
                    HloToPPHloOpConverter<stablehlo::OrOp>,
                    HloToPPHloOpConverter<stablehlo::PadOp>,
                    HloToPPHloOpConverter<stablehlo::PopulationCountOp>,
                    HloToPPHloOpConverter<stablehlo::PowOp>,
                    HloToPPHloOpConverter<stablehlo::ReduceOp>,
                    HloToPPHloOpConverter<stablehlo::ReduceWindowOp>,
                    HloToPPHloOpConverter<stablehlo::RemOp>,
                    HloToPPHloOpConverter<stablehlo::ReverseOp>,
                    HloToPPHloOpConverter<stablehlo::ReshapeOp>,
                    HloToPPHloOpConverter<stablehlo::RealOp>,
                    HloToPPHloOpConverter<stablehlo::ReturnOp>,
                    HloToPPHloOpConverter<stablehlo::RngOp>,
                    HloToPPHloOpConverter<stablehlo::RoundOp>,
                    HloToPPHloOpConverter<stablehlo::RoundNearestEvenOp>,
                    HloToPPHloOpConverter<stablehlo::RsqrtOp>,
                    HloToPPHloOpConverter<stablehlo::SineOp>,
                    HloToPPHloOpConverter<stablehlo::SelectOp>,
                    HloToPPHloOpConverter<stablehlo::SelectAndScatterOp>,
                    HloToPPHloOpConverter<stablehlo::ShiftLeftOp>,
                    HloToPPHloOpConverter<stablehlo::ShiftRightArithmeticOp>,
                    HloToPPHloOpConverter<stablehlo::ShiftRightLogicalOp>,
                    HloToPPHloOpConverter<stablehlo::SignOp>,
                    HloToPPHloOpConverter<stablehlo::SliceOp>,
                    HloToPPHloOpConverter<stablehlo::SortOp>,
                    HloToPPHloOpConverter<stablehlo::SqrtOp>,
                    HloToPPHloOpConverter<stablehlo::SubtractOp>,
                    HloToPPHloOpConverter<stablehlo::TanhOp>,
                    HloToPPHloOpConverter<stablehlo::TransposeOp>,
                    HloToPPHloOpConverter<stablehlo::WhileOp>,
                    HloToPPHloOpConverter<stablehlo::XorOp>>(converter, context,
                                                             vis_map);
  }

 public:
  HloLegalizeToPPHlo(const HloLegalizeToPPHlo &) = default;
  HloLegalizeToPPHlo() = default;

  void runOnOperation() override {
    // Stage 1: Run a visibility discover pass to tag all Values' visibility
    ValueVisibilityMap vis_map =
        VisibilityDiscovery(input_vis_list_, getOperation());

    auto &context = getContext();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    HloToPPHloTypeConverter converter(&context);

    // To pphlo dialect, ModuleOp is also a thing that we won't handle.
    target.addLegalDialect<PPHloDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::UnrealizedConversionCastOp>();
    // After conversion, there shouldn't be any mhlo dialect thingy left.
    target.addIllegalDialect<stablehlo::StablehloDialect>();

    TypeTools typetools(&getContext());
    auto is_func_sig_legal = [&](FunctionType ftype) {
      for (int64_t idx = 0; idx < ftype.getNumInputs(); ++idx) {
        auto actual_vis = typetools.getTypeVisibility(ftype.getInput(idx));
        if (actual_vis != vis_map.getInputsVisibility(idx)) {
          return false;
        }
      }

      for (int64_t idx = 0; idx < ftype.getNumResults(); ++idx) {
        auto actual_vis = typetools.getTypeVisibility(ftype.getResult(idx));
        if (actual_vis != vis_map.getOutputVisibility(idx)) {
          return false;
        }
      }

      return true;
    };

    // FcnOp is only legitimate iff signature and body is legal
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return is_func_sig_legal(op.getFunctionType()) &&
                 converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });

    auto is_return_op_legal = [&](mlir::func::ReturnOp op) {
      for (int64_t idx = 0; idx < op.getNumOperands(); ++idx) {
        auto actual_vis =
            typetools.getTypeVisibility(op.getOperandTypes()[idx]);
        if (actual_vis != vis_map.getOutputVisibility(idx)) {
          return false;
        }
      }
      return true;
    };

    // We keep mlir return op legal here.
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return converter.isLegal(op.getOperandTypes()) &&
                 is_return_op_legal(op);
        });

    // Stage 2: Do an actual dialect conversion.
    populateHLOToPPHloConversionPattern(converter, patterns, vis_map);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToPPHloPass() {
  return std::make_unique<HloLegalizeToPPHlo>();
}

}  // namespace mlir::spu::pphlo
