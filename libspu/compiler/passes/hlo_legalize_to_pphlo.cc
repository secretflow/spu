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

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/passes/map_stablehlo_to_pphlo_op.h"
#include "libspu/compiler/passes/pass_details.h"
#include "libspu/compiler/passes/value_visibility_map.h"
#include "libspu/compiler/passes/visibility_inference.h"
#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo_attrs.h"
#include "libspu/dialect/pphlo_base_enums.h"
#include "libspu/dialect/pphlo_ops.h"
#include "libspu/dialect/pphlo_types.h"

namespace mlir::pphlo {
namespace {

ValueVisibilityMap
VisibilityDiscovery(const llvm::ArrayRef<std::string> input_vis_list,
                    ModuleOp op) {
  // Get the main function
  auto entry_func = op.lookupSymbol<mlir::func::FuncOp>("main");

  SPU_ENFORCE(entry_func != nullptr, "Cannot find main entry point");

  ValueVisibilityMap vis_map;
  // Populate top level io visibility
  for (const auto &blockargs : entry_func.getBody().getArguments()) {
    SPU_ENFORCE(blockargs.getArgNumber() < input_vis_list.size(),
                "Input visibility list does not match actual inputs.");
    auto v =
        symbolizeEnum<Visibility>(input_vis_list[blockargs.getArgNumber()]);
    SPU_ENFORCE(v.has_value(), "Input visibility list has invalid value.");
    vis_map.setValueVisibility(blockargs, *v);
  }

  VisibilityInference inference(vis_map);
  inference.inferFunc(entry_func);

  return vis_map;
}

TypeTools typetools_;
/// Type converter for mhlo type to pphlo types
class HloToPPHloTypeConverter : public TypeConverter {
private:
  static Type convertRankedTensorType(RankedTensorType type) {
    Type oriElmTy = type.getElementType();
    Type newElmTy;
    if (oriElmTy.isa<::mlir::FloatType>() ||
        oriElmTy.isa<::mlir::IntegerType>()) {
      newElmTy = ::mlir::pphlo::UnsetType::get(oriElmTy);
    } else {
      newElmTy = oriElmTy;
    }
    return RankedTensorType::get(type.getShape(), newElmTy);
  }

  static Value materializeToMPCTensor(OpBuilder &builder, RankedTensorType type,
                                      ValueRange inputs, Location loc) {
    SPU_ENFORCE(inputs.size() == 1);
    SPU_ENFORCE(inputs[0].getType().isa<RankedTensorType>());

    // To unknown type is always a noop, just forward operands
    if (typetools_.isMPCType<UnsetType>(type)) {
      return inputs.front();
    }

    // Deferred materialization
    auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0]);

    return op.getResults()[0];
  }

public:
  HloToPPHloTypeConverter() {
    // Keep all types unchanged.
    addConversion([&](RankedTensorType type) -> Type {
      return convertRankedTensorType(type);
    });
    addTargetMaterialization(materializeToMPCTensor);
  }

  static Type getTypeWithVisibility(Type type, Visibility vis) {
    return typetools_.getTypeWithVisibility(type, vis);
  }
};

Visibility getOperandVisibility(const mlir::Value &v) {
  if (typetools_.isMPCType<UnsetType>(v.getType())) {
    if (auto dop = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      for (const auto &result : llvm::enumerate(dop.getResults())) {
        if (result.value() == v) {
          return typetools_.getTypeVisibility(
              dop->getOperandTypes()[result.index()]);
        }
      }
    }
    llvm_unreachable("Should not hit here.");
  }
  return typetools_.getTypeVisibility(v.getType());
}

class FuncOpConverter : public OpConversionPattern<::mlir::func::FuncOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  FuncOpConverter(TypeConverter &type_converter, MLIRContext *context,
                  const ValueVisibilityMap &vis)
      : OpConversionPattern<::mlir::func::FuncOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(::mlir::func::FuncOp op,
                  ::mlir::func::FuncOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);

    auto functionType = op.getFunctionType();
    auto &region = op.getBody();

    // Convert non-entry blocks
    SmallVector<TypeConverter::SignatureConversion, 2> conversions;
    for (Block &block : llvm::drop_begin(region, 1)) {
      conversions.emplace_back(block.getNumArguments());
      TypeConverter::SignatureConversion &back = conversions.back();
      for (BlockArgument blockArgument : block.getArguments()) {
        auto idx = blockArgument.getArgNumber();
        auto vis_v = vis_.getValueVisibility(blockArgument);
        auto convertedType = HloToPPHloTypeConverter::getTypeWithVisibility(
            typeConverter->convertType(blockArgument.getType()), vis_v);

        back.addInputs(idx, convertedType);
      }
    }

    if (failed(rewriter.convertNonEntryRegionTypes(&region, *typeConverter,
                                                   conversions))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(functionType.getNumInputs());
    for (const auto &blockarg : llvm::enumerate(op.getBody().getArguments())) {
      auto vis_v = vis_.getValueVisibility(blockarg.value());
      auto convertedType = HloToPPHloTypeConverter::getTypeWithVisibility(
          typeConverter->convertType(blockarg.value().getType()), vis_v);
      conversion.addInputs(blockarg.index(), convertedType);
    }

    // If the SignatureConversion doesn't apply, bail out.
    if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter(),
                                           &conversion))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    // Update the signature of the function.
    SmallVector<Type, 2> newResultTypes;
    if (failed(typeConverter->convertTypes(functionType.getResults(),
                                           newResultTypes))) {
      rewriter.cancelRootUpdate(op);
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
          HloToPPHloTypeConverter::getTypeWithVisibility(resultType.value(),
                                                         vis_v);
    }
    op.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                        newResultTypes));
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

class ReturnOpConverter : public OpConversionPattern<::mlir::func::ReturnOp> {
public:
  ReturnOpConverter(TypeConverter &type_converter, MLIRContext *context,
                    const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<::mlir::func::ReturnOp>(type_converter, context) {}

  LogicalResult
  matchAndRewrite(::mlir::func::ReturnOp op,
                  ::mlir::func::ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    rewriter.updateRootInPlace(
        op, [&]() { operation->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class HloCompToPPHloOpConverter
    : public OpConversionPattern<stablehlo::CompareOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloCompToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                            const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::CompareOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::CompareOp hlo_op,
                  stablehlo::CompareOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hlo_op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()), result_vis);

    auto comp_direction = hlo_op.getComparisonDirection();

    SmallVector<Value, 2> operands(adaptor.getOperands());

    if (comp_direction == stablehlo::ComparisonDirection::EQ) {
      rewriter.replaceOpWithNewOp<pphlo::EqualOp>(hlo_op, resultType, operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::NE) {
      rewriter.replaceOpWithNewOp<pphlo::NotEqualOp>(hlo_op, resultType,
                                                     operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::LT) {
      rewriter.replaceOpWithNewOp<pphlo::LessOp>(hlo_op, resultType, operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::LE) {
      rewriter.replaceOpWithNewOp<pphlo::LessEqualOp>(hlo_op, resultType,
                                                      operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::GT) {
      rewriter.replaceOpWithNewOp<pphlo::GreaterOp>(hlo_op, resultType,
                                                    operands);
    } else if (comp_direction == stablehlo::ComparisonDirection::GE) {
      rewriter.replaceOpWithNewOp<pphlo::GreaterEqualOp>(hlo_op, resultType,
                                                         operands);
    } else {
      return failure();
    }
    return success();
  }
};

template <typename HloReduceOpTy>
struct ReduceOpConverter : public OpConversionPattern<HloReduceOpTy> {
private:
  const ValueVisibilityMap &vis_;

public:
  ReduceOpConverter(TypeConverter &type_converter, MLIRContext *context,
                    const ValueVisibilityMap &vis)
      : OpConversionPattern<HloReduceOpTy>(type_converter, context), vis_(vis) {
  }

  LogicalResult
  matchAndRewrite(HloReduceOpTy op,
                  typename ReduceOpConverter::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // We may need to materialize operands
    llvm::SmallVector<Value> materialized_operands;
    llvm::SmallVector<Type> result_types;
    size_t num_results = op.getNumResults();

    materialized_operands.resize(2 * num_results);
    result_types.resize(num_results);

    OpBuilder builder(op);

    auto materialize = [&, this](size_t idx) {
      auto current_vis = getOperandVisibility(adaptor.getOperands()[idx]);
      auto expected_vis =
          vis_.getValueVisibility(op.getBody().getArguments()[idx]);

      if (expected_vis == current_vis) {
        materialized_operands[idx] = adaptor.getOperands()[idx];
      } else {
        auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
            adaptor.getOperands()[idx].getType(), expected_vis);
        materialized_operands[idx] =
            this->getTypeConverter()->materializeTargetConversion(
                builder, op.getLoc(), new_type, adaptor.getOperands()[idx]);
      }
    };

    for (size_t idx = 0; idx < num_results; ++idx) {
      auto result_vis = vis_.getValueVisibility(op.getResult(idx));
      // Check input vis
      materialize(idx);
      materialize(idx + num_results);
      // Push result type
      result_types[idx] = HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(op.getType(idx)), result_vis);
    }

    // Convert the region signature.
    auto &entry_block = op.getBody().front();
    TypeConverter::SignatureConversion sig_conversion(
        entry_block.getNumArguments());

    for (const auto &arg : entry_block.getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    auto new_op =
        rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<HloReduceOpTy>>(
            op, result_types, materialized_operands, op->getAttrs());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().end());

    if (failed(rewriter.convertRegionTypes(
            &new_op.getBody(), *this->getTypeConverter(), &sig_conversion))) {
      return failure();
    }

    return success();
  }
};

struct IfOpConverter : public OpConversionPattern<stablehlo::IfOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  IfOpConverter(TypeConverter &type_converter, MLIRContext *context,
                const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::IfOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::IfOp op, stablehlo::IfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<Type, 4> resultTypes;
    {
      for (const auto &ret : op->getResults()) {
        auto result_vis = vis_.getValueVisibility(ret);
        resultTypes.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
            this->getTypeConverter()->convertType(ret.getType()), result_vis));
      }
    }

    // Convert true region signature.
    auto &true_region = op.getTrueBranch();
    TypeConverter::SignatureConversion true_sig_conversion(
        true_region.getNumArguments());

    for (const auto &arg : true_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      true_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Convert false region signature.
    auto &false_region = op.getFalseBranch();
    TypeConverter::SignatureConversion false_sig_conversion(
        false_region.getNumArguments());

    for (const auto &arg : false_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      false_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    llvm::SmallVector<Value, 2> operands(adaptor.getOperands());

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::IfOp>(
        op, resultTypes, operands, op->getAttrs());

    // Copy over the operations inside true/false region.
    rewriter.inlineRegionBefore(op.getTrueBranch(), new_op.getTrueBranch(),
                                new_op.getTrueBranch().end());
    rewriter.inlineRegionBefore(op.getFalseBranch(), new_op.getFalseBranch(),
                                new_op.getFalseBranch().end());

    if (failed(rewriter.convertRegionTypes(&new_op.getTrueBranch(),
                                           *getTypeConverter(),
                                           &true_sig_conversion))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&new_op.getFalseBranch(),
                                           *getTypeConverter(),
                                           &false_sig_conversion))) {
      return failure();
    }

    return success();
  }
};

struct CaseOpConverter : public OpConversionPattern<stablehlo::CaseOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  CaseOpConverter(TypeConverter &type_converter, MLIRContext *context,
                  const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::CaseOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::CaseOp op, stablehlo::CaseOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<Type, 4> resultTypes;
    {
      for (const auto &ret : op->getResults()) {
        auto result_vis = vis_.getValueVisibility(ret);
        resultTypes.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
            this->getTypeConverter()->convertType(ret.getType()), result_vis));
      }
    }

    // Create new op
    llvm::SmallVector<Value, 2> operands(adaptor.getOperands());

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::CaseOp>(
        op, resultTypes, operands, op->getAttrs(), op.getNumRegions());

    // Convert each region
    llvm::SmallVector<TypeConverter::SignatureConversion, 2> sig_converters;
    for (auto &r : op.getBranches()) {
      TypeConverter::SignatureConversion sig_conversion(r.getNumArguments());
      for (const auto &arg : r.getArguments()) {
        auto arg_t = getTypeConverter()->convertType(arg.getType());
        auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
            arg_t, vis_.getValueVisibility(arg));
        sig_conversion.addInputs(arg.getArgNumber(), lower_t);
      }
      sig_converters.emplace_back(std::move(sig_conversion));
    }

    // Copy over the operations inside each region.
    for (const auto &r : llvm::enumerate(op.getBranches())) {
      rewriter.inlineRegionBefore(r.value(), new_op.getBranches()[r.index()],
                                  new_op.getBranches()[r.index()].end());
    }

    // Convert each region type
    for (const auto &r : llvm::enumerate(new_op.getBranches())) {
      if (failed(rewriter.convertRegionTypes(&r.value(), *getTypeConverter(),
                                             &sig_converters[r.index()]))) {
        return failure();
      }
    }

    return success();
  }
};

struct WhileOpConverter : public OpConversionPattern<stablehlo::WhileOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  WhileOpConverter(TypeConverter &type_converter, MLIRContext *context,
                   const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::WhileOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::WhileOp op, stablehlo::WhileOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type, 4> resultTypes;
    {
      for (const auto &ret : op->getResults()) {
        auto result_vis = vis_.getValueVisibility(ret);
        resultTypes.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
            this->getTypeConverter()->convertType(ret.getType()), result_vis));
      }
    }

    // Convert cond region signature.
    auto &cond_region = op.getCond();
    TypeConverter::SignatureConversion cond_sig_conversion(
        cond_region.getNumArguments());

    for (const auto &arg : cond_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      cond_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Convert body region signature.
    auto &body_region = op.getBody();
    TypeConverter::SignatureConversion body_sig_conversion(
        body_region.getNumArguments());

    for (const auto &arg : body_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      body_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // May need to materialize operands
    OpBuilder builder(op);
    llvm::SmallVector<Value, 6> operands(adaptor.getOperands());
    llvm::SmallVector<Value, 6> materializedOperands;
    for (const auto &operand : llvm::enumerate(operands)) {
      auto currentVis = getOperandVisibility(operand.value());
      auto targetVis =
          vis_.getValueVisibility(op.getBody().getArgument(operand.index()));
      if (currentVis == targetVis) {
        materializedOperands.emplace_back(operand.value());
      } else {
        auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
            operand.value().getType(), targetVis);
        materializedOperands.emplace_back(
            getTypeConverter()->materializeTargetConversion(
                builder, op->getLoc(), new_type, operand.value()));
      }
    }

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::WhileOp>(
        op, resultTypes, materializedOperands, op->getAttrs());

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

    return success();
  }
};

template <typename HloOpTy>
class HloToPPHloOpConverter : public OpConversionPattern<HloOpTy> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<HloOpTy>(type_converter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(HloOpTy hlo_op,
                  typename HloToPPHloOpConverter::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hlo_op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<HloOpTy>>(
        hlo_op, resultType, adaptor.getOperands(), hlo_op->getAttrs());

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

  LogicalResult
  matchAndRewrite(stablehlo::ConstantOp hlo_op,
                  stablehlo::ConstantOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<stablehlo::ConstantOp>>(
        hlo_op, hlo_op.getValue());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::IotaOp>
    : public OpConversionPattern<stablehlo::IotaOp> {
public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<stablehlo::IotaOp>(type_converter, context) {}

  LogicalResult
  matchAndRewrite(stablehlo::IotaOp hlo_op,
                  stablehlo::IotaOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()),
        Visibility::VIS_PUBLIC);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<stablehlo::IotaOp>>(
        hlo_op, resultType, hlo_op.getIotaDimension());
    return success();
  }
};

/// Need a special conversion rule for Dot to drop precision configs
template <>
class HloToPPHloOpConverter<stablehlo::DotOp>
    : public OpConversionPattern<stablehlo::DotOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::DotOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::DotOp hlo_op, stablehlo::DotOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hlo_op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<stablehlo::DotOp>>(
        hlo_op, resultType, adaptor.getOperands());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::DotGeneralOp>
    : public OpConversionPattern<stablehlo::DotGeneralOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::DotGeneralOp>(type_converter, context),
        vis_(vis) {}

  static Value ensureAtLeast3D(ConversionPatternRewriter &rewriter,
                               Value operand, Visibility expected_vis) {
    auto type = operand.getType().dyn_cast<RankedTensorType>();
    if (type.getRank() >= 3) {
      return operand;
    }

    std::vector<int64_t> new_shape(type.getShape());
    new_shape.emplace_back(1); // Add a trailing one dimension
    auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
        RankedTensorType::get(new_shape, type.getElementType()), expected_vis);
    return rewriter.create<pphlo::ReshapeOp>(operand.getLoc(), new_type,
                                             operand);
  }

  LogicalResult
  matchAndRewrite(stablehlo::DotGeneralOp hlo_op,
                  stablehlo::DotGeneralOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hlo_op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()), result_vis);

    auto attr = DotDimensionNumbersAttr::get(
        hlo_op->getContext(),
        hlo_op.getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
        hlo_op.getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
        hlo_op.getDotDimensionNumbersAttr().getLhsContractingDimensions(),
        hlo_op.getDotDimensionNumbersAttr().getRhsContractingDimensions());

    rewriter.replaceOpWithNewOp<pphlo::DotGeneralOp>(
        hlo_op, resultType,
        ensureAtLeast3D(rewriter, adaptor.getLhs(),
                        vis_.getValueVisibility(hlo_op.getLhs())),
        ensureAtLeast3D(rewriter, adaptor.getRhs(),
                        vis_.getValueVisibility(hlo_op.getRhs())),
        attr);
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::ReturnOp>
    : public OpConversionPattern<stablehlo::ReturnOp> {
public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ReturnOp>(type_converter, context) {}

  LogicalResult
  matchAndRewrite(stablehlo::ReturnOp op, stablehlo::ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expected vis
    auto *owning_region = op->getParentRegion();
    auto *owning_op = owning_region->getParentOp();

    // The owning op should already been lowered..so it's pphlo if/case
    if (llvm::isa<pphlo::IfOp>(owning_op) ||
        llvm::isa<pphlo::CaseOp>(owning_op)) {
      // Has multiple regions, need unify
      SPU_ENFORCE(owning_op, "mhlo::return should not occur in top region");

      TypeTools tools;

      OpBuilder builder(op);
      auto materialize = [&, this](const mlir::Value &operand,
                                   Visibility except_vis) {
        auto current_vis = getOperandVisibility(operand);
        if (except_vis == current_vis) {
          return operand;
        } else {
          auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
              operand.getType(), except_vis);
          return this->getTypeConverter()->materializeTargetConversion(
              builder, op.getLoc(), new_type, operand);
        }
      };

      llvm::SmallVector<Value, 2> materialized;

      SPU_ENFORCE(adaptor.getOperands().size() == owning_op->getNumResults(),
                  "adaptor has {} operands while owning op has {} results",
                  adaptor.getOperands().size(), owning_op->getNumResults());

      for (const auto &r : llvm::enumerate(owning_op->getResults())) {
        auto expected_vis = tools.getTypeVisibility(r.value().getType());
        materialized.emplace_back(
            materialize(adaptor.getOperands()[r.index()], expected_vis));
      }

      rewriter.replaceOpWithNewOp<pphlo::ReturnOp>(op, std::nullopt,
                                                   materialized);
    } else {
      rewriter.replaceOpWithNewOp<pphlo::ReturnOp>(op, std::nullopt,
                                                   adaptor.getOperands());
    }

    return success();
  }
};

template <>
struct HloToPPHloOpConverter<stablehlo::SelectAndScatterOp>
    : public OpConversionPattern<stablehlo::SelectAndScatterOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::SelectAndScatterOp>(type_converter,
                                                           context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::SelectAndScatterOp op,
                  stablehlo::SelectAndScatterOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // We may need to materialize operands
    OpBuilder builder(op);

    // Select
    auto materialize = [&, this](const mlir::Value &operand,
                                 Visibility except_vis) {
      auto current_vis = getOperandVisibility(operand);
      if (except_vis == current_vis) {
        return operand;
      } else {
        auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
            operand.getType(), except_vis);
        return this->getTypeConverter()->materializeTargetConversion(
            builder, op.getLoc(), new_type, operand);
      }
    };

    auto promoted_vis = mlir::pphlo::TypeTools::inferResultVisibility(
        {vis_.getValueVisibility(op.getOperand()),
         vis_.getValueVisibility(op.getInitValue())});
    auto materialized_operand = materialize(adaptor.getOperand(), promoted_vis);
    auto materialized_init_value =
        materialize(adaptor.getInitValue(), promoted_vis);

    auto result_type = HloToPPHloTypeConverter::getTypeWithVisibility(
        op.getType(), vis_.getValueVisibility(op.getResult()));

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::SelectAndScatterOp>(
        op, result_type, materialized_operand, adaptor.getSource(),
        materialized_init_value, op.getWindowDimensionsAttr(),
        op.getWindowStridesAttr(), op.getPaddingAttr());

    // Convert the region signature.
    TypeConverter::SignatureConversion select_sig_conversion(
        op.getSelect().front().getNumArguments());

    for (const auto &arg : op.getSelect().front().getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      select_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    TypeConverter::SignatureConversion scatter_sig_conversion(
        op.getScatter().front().getNumArguments());

    for (const auto &arg : op.getScatter().front().getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
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
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::RngOp>
    : public OpConversionPattern<stablehlo::RngOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::RngOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::RngOp op, stablehlo::RngOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::RngOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::SortOp>
    : public OpConversionPattern<stablehlo::SortOp> {
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::SortOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::SortOp op, stablehlo::SortOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto comp_ret = llvm::dyn_cast<stablehlo::ReturnOp>(
        op.getComparator().back().getTerminator());
    SPU_ENFORCE(comp_ret.getNumOperands() == 1,
                "SortOp comparator can only return one value");

    llvm::SmallVector<Type, 2> ret_types;
    for (const auto &ret : op->getResults()) {
      auto ret_vis = vis_.getValueVisibility(ret);
      ret_types.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(ret.getType()), ret_vis));
    }

    // Convert the region signature.
    auto &comp_region = op.getComparator();
    TypeConverter::SignatureConversion sig_conversion(
        comp_region.getNumArguments());

    for (const auto &arg : comp_region.getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // materialize inputs
    auto materialize = [&](mlir::Value v, Visibility vis) {
      auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(v.getType()), vis);
      return this->getTypeConverter()->materializeTargetConversion(
          rewriter, op.getLoc(), new_type, v);
    };
    llvm::SmallVector<mlir::Value, 2> new_operands;
    for (int64_t idx = 0; idx < op->getNumOperands(); ++idx) {
      auto ret_vis = vis_.getValueVisibility(op.getResult(idx));
      auto operand_vis = vis_.getValueVisibility(op->getOperand(idx));
      if (ret_vis == operand_vis) {
        new_operands.emplace_back(adaptor.getOperands()[idx]);
      } else {
        new_operands.emplace_back(
            materialize(adaptor.getOperands()[idx], ret_vis));
      }
    }

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::SortOp>(
        op, ret_types, new_operands, op.getDimension(), op.getIsStable());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.getComparator(), new_op.getComparator(),
                                new_op.getComparator().end());

    if (failed(rewriter.convertRegionTypes(&new_op.getComparator(),
                                           *this->getTypeConverter(),
                                           &sig_conversion))) {
      return failure();
    }

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::GatherOp>
    : public OpConversionPattern<stablehlo::GatherOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::GatherOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::GatherOp op, stablehlo::GatherOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto old_attr = op.getDimensionNumbers();
    pphlo::GatherDimensionNumbersAttr attr = GatherDimensionNumbersAttr::get(
        op.getContext(), old_attr.getOffsetDims(),
        old_attr.getCollapsedSliceDims(), old_attr.getStartIndexMap(),
        old_attr.getIndexVectorDim());

    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::GatherOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1],
        attr, op.getSliceSizes(), op.getIndicesAreSorted());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::ConvolutionOp>
    : public OpConversionPattern<stablehlo::ConvolutionOp> {
private:
  const ValueVisibilityMap &vis_;

  /// Returns true if the given `attr` is a splat of the given `value`.
  static bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
    return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
  }

  // Apply dilation and padding to the input of a convolution.
  static Value applyConvolutionPadding(Location loc, Value input,
                                       DenseIntElementsAttr padding,
                                       DenseIntElementsAttr lhs_dilation,
                                       llvm::ArrayRef<int64_t> dim_mappings,
                                       OpBuilder &rewriter) {
    if ((!padding || isSplatValue(padding, 0)) &&
        (!lhs_dilation || isSplatValue(lhs_dilation, 1))) {
      return input;
    }

    auto inputType = input.getType().cast<ShapedType>();
    auto rank = inputType.getRank();

    // Translate window padding into low/high padding.
    SmallVector<int64_t, 8> padLow(rank, 0);
    SmallVector<int64_t, 8> padHigh(rank, 0);
    if (padding) {
      // The padding attribute contains two values per dimension, but excludes
      // the batch and feature dimensions.
      assert(
          rank * 2 == padding.size() + 4 &&
          "There should be 2 padding values per dimension, i.e low and high.");
      for (auto i : llvm::seq<int64_t>(0, padding.size() / 2)) {
        auto dim = dim_mappings[i];
        padLow[dim] = padding.getValues<int64_t>()[i * 2];
        padHigh[dim] = padding.getValues<int64_t>()[i * 2 + 1];
      }
    }

    // Translate input dilation into interior padding.
    SmallVector<int64_t, 8> padInterior(rank, 0);
    if (lhs_dilation) {
      assert(rank == lhs_dilation.size() + 2);
      for (auto i : llvm::seq<int64_t>(0, lhs_dilation.size())) {
        auto dim = dim_mappings[i];
        padInterior[dim] = lhs_dilation.getValues<int64_t>()[i] - 1;
      }
    }

    TypeTools type_tools;
    auto indexType = rewriter.getIntegerType(64);
    auto attrType = RankedTensorType::get({rank}, indexType);
    Value zero = rewriter.create<pphlo::ConstantOp>(
        loc, rewriter.getZeroAttr(RankedTensorType::get(
                 {}, type_tools.getExpressedType(inputType.getElementType()))));
    zero = rewriter.create<pphlo::ConvertOp>(
        loc, RankedTensorType::get({}, inputType.getElementType()), zero);
    return rewriter.create<pphlo::PadOp>(
        loc, input, zero, DenseIntElementsAttr::get(attrType, padLow),
        DenseIntElementsAttr::get(attrType, padHigh),
        DenseIntElementsAttr::get(attrType, padInterior));
  }

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ConvolutionOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::ConvolutionOp op,
                  stablehlo::ConvolutionOpAdaptor adaptor,
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

    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    auto materialize = [&](mlir::Value v, Visibility vis) {
      auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(v.getType()), vis);
      return this->getTypeConverter()->materializeTargetConversion(
          rewriter, op.getLoc(), new_type, v);
    };

    auto modifiedLhs =
        materialize(adaptor.getLhs(), vis_.getValueVisibility(op.getLhs()));

    modifiedLhs = applyConvolutionPadding(
        op->getLoc(), modifiedLhs, adaptor.getPaddingAttr(),
        adaptor.getLhsDilationAttr(),
        adaptor.getDimensionNumbers().getInputSpatialDimensions(), rewriter);

    auto modifiedRhs =
        materialize(adaptor.getRhs(), vis_.getValueVisibility(op.getRhs()));

    modifiedRhs = applyConvolutionPadding(
        op.getLoc(), modifiedRhs, nullptr, adaptor.getRhsDilationAttr(),
        op.getDimensionNumbers().getKernelSpatialDimensions(), rewriter);

    auto reversals = op.getWindowReversal();
    if (reversals.has_value()) {
      llvm::SmallVector<int64_t> reversedDims;
      for (const auto &idxAndBool :
           llvm::enumerate(reversals->getValues<bool>())) {
        if (idxAndBool.value()) {
          reversedDims.push_back(
              op.getDimensionNumbers()
                  .getKernelSpatialDimensions()[idxAndBool.index()]);
        }
      }

      modifiedRhs = rewriter.create<pphlo::ReverseOp>(
          op.getLoc(), modifiedRhs,
          mlir::DenseIntElementsAttr::get(
              RankedTensorType::get(reversedDims.size(),
                                    rewriter.getIntegerType(64)),
              reversedDims));
    }

    rewriter.replaceOpWithNewOp<pphlo::ConvolutionOp>(
        op, resultType, modifiedLhs, modifiedRhs,
        op.getWindowStrides().value_or(nullptr), attr,
        op.getFeatureGroupCount(), op.getBatchGroupCount());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::PadOp>
    : public OpConversionPattern<stablehlo::PadOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::PadOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::PadOp op, stablehlo::PadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type result_type = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    llvm::SmallVector<Value, 2> materialized_operands;
    OpBuilder builder(op);
    for (const auto &old_operand : llvm::enumerate(op.getOperands())) {
      auto op_vis = vis_.getValueVisibility(old_operand.value());
      if (op_vis != result_vis) {
        Type new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
            adaptor.getOperands()[old_operand.index()].getType(), result_vis);
        materialized_operands.emplace_back(
            getTypeConverter()->materializeTargetConversion(
                builder, op.getLoc(), new_type,
                adaptor.getOperands()[old_operand.index()]));
      } else {
        materialized_operands.emplace_back(
            adaptor.getOperands()[old_operand.index()]);
      }
    }

    rewriter.replaceOpWithNewOp<pphlo::PadOp>(
        op, result_type, materialized_operands, op->getAttrs());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::BitcastConvertOp>
    : public OpConversionPattern<stablehlo::BitcastConvertOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::BitcastConvertOp>(type_converter,
                                                         context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::BitcastConvertOp op,
                  stablehlo::BitcastConvertOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    auto in_type_size = op->getOperandTypes()[0]
                            .dyn_cast<RankedTensorType>()
                            .getElementTypeBitWidth();
    auto out_type_size = op->getResultTypes()[0]
                             .dyn_cast<RankedTensorType>()
                             .getElementTypeBitWidth();

    SPU_ENFORCE(in_type_size == out_type_size,
                "BitcastConvert with different input/output element size is "
                "not supported");

    rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(
        op, resultType, adaptor.getOperands()[0]);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::ConcatenateOp>
    : public OpConversionPattern<stablehlo::ConcatenateOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::ConcatenateOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::ConcatenateOp op,
                  stablehlo::ConcatenateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type result_type = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    OpBuilder builder(op);
    SmallVector<Value, 2> materialized_operands;
    for (size_t idx = 0; idx < adaptor.getOperands().size(); ++idx) {
      auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(op->getOperand(idx).getType()),
          result_vis);
      materialized_operands.emplace_back(
          this->getTypeConverter()->materializeTargetConversion(
              builder, op.getLoc(), new_type, adaptor.getOperands()[idx]));
    }

    rewriter.replaceOpWithNewOp<pphlo::ConcatenateOp>(
        op, result_type, materialized_operands, op.getDimension());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<stablehlo::DynamicUpdateSliceOp>
    : public OpConversionPattern<stablehlo::DynamicUpdateSliceOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<stablehlo::DynamicUpdateSliceOp>(type_converter,
                                                             context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(stablehlo::DynamicUpdateSliceOp op,
                  stablehlo::DynamicUpdateSliceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type result_type = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    OpBuilder builder(op);

    auto materialize = [&](Value in) {
      auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(in.getType()), result_vis);
      return this->getTypeConverter()->materializeTargetConversion(
          builder, op.getLoc(), new_type, in);
    };

    Value materized_operand = materialize(adaptor.getOperand());
    Value materized_update = materialize(adaptor.getUpdate());

    rewriter.replaceOpWithNewOp<pphlo::DynamicUpdateSliceOp>(
        op, result_type, materized_operand, materized_update,
        adaptor.getStartIndices());

    return success();
  }
};

struct HloLegalizeToPPHlo
    : public HloLegalizeToPPHloPassBase<HloLegalizeToPPHlo> {
private:
  static void
  populateHLOToPPHloConversionPattern(HloToPPHloTypeConverter &converter,
                                      RewritePatternSet &patterns,
                                      const ValueVisibilityMap &vis_map) {
    auto *context = patterns.getContext();

    patterns.insert<
        FuncOpConverter, ReturnOpConverter, HloCompToPPHloOpConverter,
        ReduceOpConverter<stablehlo::ReduceOp>,
        ReduceOpConverter<stablehlo::ReduceWindowOp>, WhileOpConverter,
        IfOpConverter, CaseOpConverter, HloToPPHloOpConverter<stablehlo::AbsOp>,
        HloToPPHloOpConverter<stablehlo::AddOp>,
        HloToPPHloOpConverter<stablehlo::AndOp>,
        HloToPPHloOpConverter<stablehlo::BitcastConvertOp>,
        HloToPPHloOpConverter<stablehlo::BroadcastInDimOp>,
        HloToPPHloOpConverter<stablehlo::CeilOp>,
        HloToPPHloOpConverter<stablehlo::ClampOp>,
        HloToPPHloOpConverter<stablehlo::ConcatenateOp>,
        HloToPPHloOpConverter<stablehlo::ConstantOp>,
        HloToPPHloOpConverter<stablehlo::ConvertOp>,
        HloToPPHloOpConverter<stablehlo::ConvolutionOp>,
        HloToPPHloOpConverter<stablehlo::DivOp>,
        HloToPPHloOpConverter<stablehlo::DotOp>,
        HloToPPHloOpConverter<stablehlo::DotGeneralOp>,
        HloToPPHloOpConverter<stablehlo::GatherOp>,
        HloToPPHloOpConverter<stablehlo::DynamicSliceOp>,
        HloToPPHloOpConverter<stablehlo::DynamicUpdateSliceOp>,
        HloToPPHloOpConverter<stablehlo::ExpOp>,
        HloToPPHloOpConverter<stablehlo::Expm1Op>,
        HloToPPHloOpConverter<stablehlo::FloorOp>,
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
        HloToPPHloOpConverter<stablehlo::PowOp>,
        HloToPPHloOpConverter<stablehlo::RemOp>,
        HloToPPHloOpConverter<stablehlo::ReshapeOp>,
        HloToPPHloOpConverter<stablehlo::ReturnOp>,
        HloToPPHloOpConverter<stablehlo::RoundOp>,
        HloToPPHloOpConverter<stablehlo::ReverseOp>,
        HloToPPHloOpConverter<stablehlo::RngOp>,
        HloToPPHloOpConverter<stablehlo::SelectOp>,
        HloToPPHloOpConverter<stablehlo::SelectAndScatterOp>,
        HloToPPHloOpConverter<stablehlo::ShiftLeftOp>,
        HloToPPHloOpConverter<stablehlo::ShiftRightArithmeticOp>,
        HloToPPHloOpConverter<stablehlo::ShiftRightLogicalOp>,
        HloToPPHloOpConverter<stablehlo::SignOp>,
        HloToPPHloOpConverter<stablehlo::SliceOp>,
        HloToPPHloOpConverter<stablehlo::SortOp>,
        HloToPPHloOpConverter<stablehlo::SqrtOp>,
        HloToPPHloOpConverter<stablehlo::RsqrtOp>,
        HloToPPHloOpConverter<stablehlo::SubtractOp>,
        HloToPPHloOpConverter<stablehlo::TanhOp>,
        HloToPPHloOpConverter<stablehlo::TransposeOp>,
        HloToPPHloOpConverter<stablehlo::XorOp>>(converter, context, vis_map);
  }

public:
  HloLegalizeToPPHlo(const HloLegalizeToPPHlo &) = default;
  HloLegalizeToPPHlo() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    HloToPPHloTypeConverter converter;

    // To pphlo dialect, ModuleOp is also a thing that we won't handle.
    target.addLegalDialect<PPHloDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    // After conversion, there shouldn't be any mhlo dialect thingy left.
    target.addIllegalDialect<stablehlo::StablehloDialect>();

    // FcnOp is only legitimate iff signature and body is legal
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });
    // We keep mlir return op legal here.
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return converter.isLegal(op.getOperandTypes());
        });

    // Stage 1: Run a visibility discover pass to tag all Values' visibility
    ValueVisibilityMap vis_map =
        VisibilityDiscovery(input_vis_list_, getOperation());

    // Stage 2: Do an actual dialect conversion.
    populateHLOToPPHloConversionPattern(converter, patterns, vis_map);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToPPHloPass() {
  return std::make_unique<HloLegalizeToPPHlo>();
}

} // namespace mlir::pphlo
