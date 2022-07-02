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
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "yasl/base/exception.h"

#include "spu/compiler/passes/map_mhlo_to_pphlo_op.h"
#include "spu/compiler/passes/pass_details.h"
#include "spu/compiler/passes/value_visibility_map.h"
#include "spu/compiler/passes/visibility_inference.h"
#include "spu/dialect/pphlo_base_enums.h"

namespace mlir::pphlo {
namespace {

/// This struct carries information of io visibility
struct IoVisibilityInfo {
  std::vector<Visibility> inputs;

  void convertFromStrings(llvm::ArrayRef<std::string> data) {
    for (const auto &s : data) {
      const auto symbolized = symbolizeEnum<Visibility>(s);
      YASL_ENFORCE(symbolized.hasValue());
      inputs.emplace_back(*symbolized);
    }
  }

  Visibility getInputVisibility(size_t idx) const {
    if (idx >= inputs.size()) {
      return Visibility::VIS_PUBLIC;
    }
    return inputs[idx];
  }
};

ValueVisibilityMap VisibilityDiscovery(ModuleOp op,
                                       const IoVisibilityInfo &input_vis) {
  // Get the main function
  auto entry_func = op.lookupSymbol<FuncOp>("main");

  YASL_ENFORCE(entry_func != nullptr);

  ValueVisibilityMap vis_map;
  // Populate top level io visibility
  for (const auto &blockargs : entry_func.getBody().getArguments()) {
    vis_map.setValueVisibility(
        blockargs, input_vis.getInputVisibility(blockargs.getArgNumber()));
  }

  VisibilityInference inference(vis_map);
  inference.inferFunc(entry_func);

  return vis_map;
}

TypeTools typetools_;
/// Type converter for mhlo type to pphlo types
class HloToPPHloTypeConverter : public TypeConverter {
private:
  Type convertRankedTensorType(RankedTensorType type) {
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
    YASL_ENFORCE(inputs.size() == 1);
    YASL_ENFORCE(inputs[0].getType().isa<RankedTensorType>());

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
      : OpConversionPattern<::mlir::FuncOp>(type_converter, context),
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
    YASL_ENFORCE(retOp->getNumOperands() == newResultTypes.size());

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

// TODO: Add a rsqrt op if we have cases that can benefit from fused op.
class RSqrtOpConverter : public OpConversionPattern<mhlo::RsqrtOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  RSqrtOpConverter(TypeConverter &type_converter, MLIRContext *context,
                   const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::RsqrtOp>(type_converter, context), vis_(vis) {
  }

  LogicalResult
  matchAndRewrite(mhlo::RsqrtOp op, mhlo::RsqrtOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    OpBuilder builder(op);

    auto r = builder.create<pphlo::SqrtOp>(op->getLoc(), resultType,
                                           adaptor.getOperands());

    rewriter.replaceOpWithNewOp<pphlo::ReciprocalOp>(op, resultType, r);

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

class HloCompToPPHloOpConverter : public OpConversionPattern<mhlo::CompareOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloCompToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                            const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::CompareOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::CompareOp hlo_op, mhlo::CompareOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hlo_op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()), result_vis);

    auto comp_direction = hlo_op.comparison_direction();

    SmallVector<Value, 2> operands(adaptor.getOperands());

    if (comp_direction == mhlo::ComparisonDirection::EQ) {
      rewriter.replaceOpWithNewOp<pphlo::EqualOp>(hlo_op, resultType, operands);
    } else if (comp_direction == mhlo::ComparisonDirection::NE) {
      rewriter.replaceOpWithNewOp<pphlo::NotEqualOp>(hlo_op, resultType,
                                                     operands);
    } else if (comp_direction == mhlo::ComparisonDirection::LT) {
      rewriter.replaceOpWithNewOp<pphlo::LessOp>(hlo_op, resultType, operands);
    } else if (comp_direction == mhlo::ComparisonDirection::LE) {
      rewriter.replaceOpWithNewOp<pphlo::LessEqualOp>(hlo_op, resultType,
                                                      operands);
    } else if (comp_direction == mhlo::ComparisonDirection::GT) {
      rewriter.replaceOpWithNewOp<pphlo::GreaterOp>(hlo_op, resultType,
                                                    operands);
    } else if (comp_direction == mhlo::ComparisonDirection::GE) {
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
          vis_.getValueVisibility(op.body().getArguments()[idx]);

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
    auto &entry_block = op.body().front();
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
    rewriter.inlineRegionBefore(op.body(), new_op.body(), new_op.body().end());

    if (failed(rewriter.convertRegionTypes(
            &new_op.body(), *this->getTypeConverter(), &sig_conversion))) {
      return failure();
    }

    return success();
  }
};

struct IfOpConverter : public OpConversionPattern<mhlo::IfOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  IfOpConverter(TypeConverter &type_converter, MLIRContext *context,
                const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::IfOp>(type_converter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::IfOp op, mhlo::IfOpAdaptor adaptor,
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
    auto &true_region = op.true_branch();
    TypeConverter::SignatureConversion true_sig_conversion(
        true_region.getNumArguments());

    for (const auto &arg : true_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      true_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Convert false region signature.
    auto &false_region = op.false_branch();
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
    rewriter.inlineRegionBefore(op.true_branch(), new_op.true_branch(),
                                new_op.true_branch().end());
    rewriter.inlineRegionBefore(op.false_branch(), new_op.false_branch(),
                                new_op.false_branch().end());

    if (failed(rewriter.convertRegionTypes(&new_op.true_branch(),
                                           *getTypeConverter(),
                                           &true_sig_conversion))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&new_op.false_branch(),
                                           *getTypeConverter(),
                                           &false_sig_conversion))) {
      return failure();
    }

    return success();
  }
};

struct WhileOpConverter : public OpConversionPattern<mhlo::WhileOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  WhileOpConverter(TypeConverter &type_converter, MLIRContext *context,
                   const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::WhileOp>(type_converter, context), vis_(vis) {
  }

  LogicalResult
  matchAndRewrite(mhlo::WhileOp op, mhlo::WhileOpAdaptor adaptor,
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
    auto &cond_region = op.cond();
    TypeConverter::SignatureConversion cond_sig_conversion(
        cond_region.getNumArguments());

    for (const auto &arg : cond_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      cond_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Convert body region signature.
    auto &body_region = op.body();
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
          vis_.getValueVisibility(op.body().getArgument(operand.index()));
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
    rewriter.inlineRegionBefore(op.body(), new_op.body(), new_op.body().end());
    rewriter.inlineRegionBefore(op.cond(), new_op.cond(), new_op.cond().end());

    if (failed(rewriter.convertRegionTypes(&new_op.body(), *getTypeConverter(),
                                           &body_sig_conversion))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&new_op.cond(), *getTypeConverter(),
                                           &cond_sig_conversion))) {
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
class HloToPPHloOpConverter<mhlo::ConstOp>
    : public OpConversionPattern<mhlo::ConstOp> {
public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<mhlo::ConstOp>(type_converter, context) {}

  LogicalResult
  matchAndRewrite(mhlo::ConstOp hlo_op, mhlo::ConstOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<mhlo::ConstOp>>(
        hlo_op, hlo_op.value());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::IotaOp>
    : public OpConversionPattern<mhlo::IotaOp> {
public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<mhlo::IotaOp>(type_converter, context) {}

  LogicalResult
  matchAndRewrite(mhlo::IotaOp hlo_op, mhlo::IotaOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()),
        Visibility::VIS_PUBLIC);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<mhlo::IotaOp>>(
        hlo_op, resultType, hlo_op.iota_dimension());
    return success();
  }
};

/// Need a special conversion rule for Dot to drop precision configs
template <>
class HloToPPHloOpConverter<mhlo::DotOp>
    : public OpConversionPattern<mhlo::DotOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::DotOp>(type_converter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::DotOp hlo_op, mhlo::DotOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hlo_op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hlo_op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<mhlo::DotOp>>(
        hlo_op, resultType, adaptor.getOperands());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::ReturnOp>
    : public OpConversionPattern<mhlo::ReturnOp> {
public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<mhlo::ReturnOp>(type_converter, context) {}

  LogicalResult
  matchAndRewrite(mhlo::ReturnOp op, mhlo::ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::ReturnOp>(op, llvm::None,
                                                 adaptor.getOperands());
    return success();
  }
};

template <>
struct HloToPPHloOpConverter<mhlo::SelectAndScatterOp>
    : public OpConversionPattern<mhlo::SelectAndScatterOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::SelectAndScatterOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::SelectAndScatterOp op,
                  mhlo::SelectAndScatterOpAdaptor adaptor,
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

    auto promoted_vis = typetools_.inferResultVisibility(
        {vis_.getValueVisibility(op.operand()),
         vis_.getValueVisibility(op.init_value())});
    auto materialized_operand = materialize(adaptor.operand(), promoted_vis);
    auto materialized_init_value =
        materialize(adaptor.init_value(), promoted_vis);

    auto result_type = HloToPPHloTypeConverter::getTypeWithVisibility(
        op.getType(), vis_.getValueVisibility(op.getResult()));

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::SelectAndScatterOp>(
        op, result_type, materialized_operand, adaptor.source(),
        materialized_init_value, op.window_dimensionsAttr(),
        op.window_dimensionsAttr(), op.paddingAttr());

    // Convert the region signature.
    TypeConverter::SignatureConversion select_sig_conversion(
        op.select().front().getNumArguments());

    for (const auto &arg : op.select().front().getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      select_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    TypeConverter::SignatureConversion scatter_sig_conversion(
        op.scatter().front().getNumArguments());

    for (const auto &arg : op.scatter().front().getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      scatter_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.select(), new_op.select(),
                                new_op.select().end());
    rewriter.inlineRegionBefore(op.scatter(), new_op.scatter(),
                                new_op.scatter().end());

    if (failed(rewriter.convertRegionTypes(&new_op.select(),
                                           *this->getTypeConverter(),
                                           &select_sig_conversion))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&new_op.scatter(),
                                           *this->getTypeConverter(),
                                           &scatter_sig_conversion))) {
      return failure();
    }

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::RngUniformOp>
    : public OpConversionPattern<mhlo::RngUniformOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::RngUniformOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::RngUniformOp op, mhlo::RngUniformOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::RngUniformOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::SortOp>
    : public OpConversionPattern<mhlo::SortOp> {
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::SortOp>(type_converter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::SortOp op, mhlo::SortOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto comp_ret =
        llvm::dyn_cast<mhlo::ReturnOp>(op.comparator().back().getTerminator());
    YASL_ENFORCE(comp_ret.getNumOperands() == 1,
                 "SortOp comparator can only return one value");

    llvm::SmallVector<Type, 2> ret_types;
    for (const auto &ret : op->getResults()) {
      auto ret_vis = vis_.getValueVisibility(ret);
      ret_types.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(ret.getType()), ret_vis));
    }

    // Convert the region signature.
    auto &comp_region = op.comparator();
    TypeConverter::SignatureConversion sig_conversion(
        comp_region.getNumArguments());

    for (const auto &arg : comp_region.getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::SortOp>(
        op, ret_types, adaptor.getOperands(), op.dimension(), op.is_stable());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.comparator(), new_op.comparator(),
                                new_op.comparator().end());

    if (failed(rewriter.convertRegionTypes(&new_op.comparator(),
                                           *this->getTypeConverter(),
                                           &sig_conversion))) {
      return failure();
    }

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::GatherOp>
    : public OpConversionPattern<mhlo::GatherOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::GatherOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::GatherOp op, mhlo::GatherOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto old_attr = op.dimension_numbers();
    pphlo::GatherDimensionNumbersAttr attr = GatherDimensionNumbersAttr::get(
        op.getContext(), old_attr.getOffsetDims(),
        old_attr.getCollapsedSliceDims(), old_attr.getStartIndexMap(),
        old_attr.getIndexVectorDim());

    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::GatherOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1],
        attr, op.slice_sizes(), op.indices_are_sorted());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::ConvOp>
    : public OpConversionPattern<mhlo::ConvOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::ConvOp>(type_converter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::ConvOp op, mhlo::ConvOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto old_attr = op.dimension_numbers();
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

    rewriter.replaceOpWithNewOp<pphlo::ConvOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1],
        op.window_strides().getValueOr(nullptr),
        op.padding().getValueOr(nullptr), op.lhs_dilation().getValueOr(nullptr),
        op.rhs_dilation().getValueOr(nullptr), attr, op.feature_group_count(),
        op.batch_group_count());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::PadOp>
    : public OpConversionPattern<mhlo::PadOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::PadOp>(type_converter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::PadOp op, mhlo::PadOpAdaptor adaptor,
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
class HloToPPHloOpConverter<mhlo::BitcastConvertOp>
    : public OpConversionPattern<mhlo::BitcastConvertOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::BitcastConvertOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::BitcastConvertOp op,
                  mhlo::BitcastConvertOpAdaptor adaptor,
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

    YASL_ENFORCE(in_type_size == out_type_size);

    rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(
        op, resultType, adaptor.getOperands()[0], out_type_size);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::ConcatenateOp>
    : public OpConversionPattern<mhlo::ConcatenateOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &type_converter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::ConcatenateOp>(type_converter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::ConcatenateOp op, mhlo::ConcatenateOpAdaptor adaptor,
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
        op, result_type, materialized_operands, op.dimension());

    return success();
  }
};

struct HloLegalizeToPPHlo
    : public HloLegalizeToPPHloPassBase<HloLegalizeToPPHlo> {
private:
  void populateHLOToPPHloConversionPattern(HloToPPHloTypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           const ValueVisibilityMap &vis_map) {
    auto *context = patterns.getContext();

    patterns.insert<
        FuncOpConverter, ReturnOpConverter, HloCompToPPHloOpConverter,
        RSqrtOpConverter, ReduceOpConverter<mhlo::ReduceOp>,
        ReduceOpConverter<mhlo::ReduceWindowOp>, WhileOpConverter,
        IfOpConverter, HloToPPHloOpConverter<mhlo::AbsOp>,
        HloToPPHloOpConverter<mhlo::AddOp>, HloToPPHloOpConverter<mhlo::AndOp>,
        HloToPPHloOpConverter<mhlo::BitcastConvertOp>,
        HloToPPHloOpConverter<mhlo::BroadcastInDimOp>,
        HloToPPHloOpConverter<mhlo::CeilOp>,
        HloToPPHloOpConverter<mhlo::ClampOp>,
        HloToPPHloOpConverter<mhlo::ConcatenateOp>,
        HloToPPHloOpConverter<mhlo::ConstOp>,
        HloToPPHloOpConverter<mhlo::ConvertOp>,
        HloToPPHloOpConverter<mhlo::ConvOp>, HloToPPHloOpConverter<mhlo::DivOp>,
        HloToPPHloOpConverter<mhlo::DotOp>,
        HloToPPHloOpConverter<mhlo::DynamicSliceOp>,
        HloToPPHloOpConverter<mhlo::DynamicUpdateSliceOp>,
        HloToPPHloOpConverter<mhlo::ExpOp>,
        HloToPPHloOpConverter<mhlo::FloorOp>,
        HloToPPHloOpConverter<mhlo::GatherOp>,
        HloToPPHloOpConverter<mhlo::IotaOp>, HloToPPHloOpConverter<mhlo::LogOp>,
        HloToPPHloOpConverter<mhlo::Log1pOp>,
        HloToPPHloOpConverter<mhlo::LogisticOp>,
        HloToPPHloOpConverter<mhlo::MaxOp>, HloToPPHloOpConverter<mhlo::MinOp>,
        HloToPPHloOpConverter<mhlo::MulOp>, HloToPPHloOpConverter<mhlo::NegOp>,
        HloToPPHloOpConverter<mhlo::NotOp>, HloToPPHloOpConverter<mhlo::OrOp>,
        HloToPPHloOpConverter<mhlo::PadOp>, HloToPPHloOpConverter<mhlo::PowOp>,
        HloToPPHloOpConverter<mhlo::RemOp>,
        HloToPPHloOpConverter<mhlo::ReshapeOp>,
        HloToPPHloOpConverter<mhlo::ReturnOp>,
        HloToPPHloOpConverter<mhlo::ReverseOp>,
        HloToPPHloOpConverter<mhlo::RngUniformOp>,
        HloToPPHloOpConverter<mhlo::SelectOp>,
        HloToPPHloOpConverter<mhlo::SelectAndScatterOp>,
        HloToPPHloOpConverter<mhlo::ShiftLeftOp>,
        HloToPPHloOpConverter<mhlo::ShiftRightArithmeticOp>,
        HloToPPHloOpConverter<mhlo::ShiftRightLogicalOp>,
        HloToPPHloOpConverter<mhlo::SliceOp>,
        HloToPPHloOpConverter<mhlo::SortOp>,
        HloToPPHloOpConverter<mhlo::SqrtOp>, HloToPPHloOpConverter<mhlo::SubOp>,
        HloToPPHloOpConverter<mhlo::TanhOp>,
        HloToPPHloOpConverter<mhlo::TransposeOp>,
        HloToPPHloOpConverter<mhlo::XorOp>>(converter, context, vis_map);
  }

  IoVisibilityInfo vis_info_;

  void parseVisibilityString() {
    if (io_visibility_json_.empty()) {
      return;
    }
    llvm::raw_os_ostream os(std::cout);
    if (auto json_v = llvm::json::parse(io_visibility_json_)) {
      llvm::json::Path::Root r;
      llvm::json::ObjectMapper map(*json_v, r);
      std::vector<std::string> str_vis;
      if (map && map.map("inputs", str_vis)) {
        vis_info_.convertFromStrings(str_vis);
      } else {
        r.printErrorContext(*json_v, os);
      }
    } else {
      handleAllErrors(json_v.takeError(), [&](const llvm::ErrorInfoBase &E) {
        os << "Failed to parse visibility JSON >>> " << io_visibility_json_
           << " <<<: " << E.message();
      });
    }
    os.flush();
  }

public:
  HloLegalizeToPPHlo(const HloLegalizeToPPHlo &) = default;
  HloLegalizeToPPHlo() = default;
  explicit HloLegalizeToPPHlo(const std::string &io_visibility_json) {
    io_visibility_json_ = io_visibility_json;
  }

  void runOnOperation() override {
    // This is a must step for cli workflow
    parseVisibilityString();

    auto &context = getContext();

    context.getTypeUniquer();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    HloToPPHloTypeConverter converter;

    // To pphlo dialect, ModuleOp is also a thing that we won't handle.
    target.addLegalDialect<PPHloDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    // After conversion, there shouldn't be any mhlo dialect thingy left.
    target.addIllegalDialect<mhlo::MhloDialect>();

    // FcnOp is only legitimate iff signature and body is legal
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    // We keep mlir return op legal here.
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return converter.isLegal(op.getOperandTypes());
        });

    // Stage 1: Run a visibility discover pass to tag all Values' visibility
    ValueVisibilityMap vis_map = VisibilityDiscovery(getOperation(), vis_info_);

    // Stage 2: Do an actual dialect conversion.
    populateHLOToPPHloConversionPattern(converter, patterns, vis_map);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeToPPHloPass(const std::string &io_visibility_json) {
  return std::make_unique<HloLegalizeToPPHlo>(io_visibility_json);
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToPPHloPass() {
  return std::make_unique<HloLegalizeToPPHlo>();
}

} // namespace mlir::pphlo
