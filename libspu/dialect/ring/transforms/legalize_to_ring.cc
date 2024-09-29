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
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// depending dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/dialect.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/ring/IR/ops.h"
#include "libspu/dialect/ring/IR/types.h"
#include "libspu/dialect/ring/transforms/map_pphlo_to_ring.h"
#include "libspu/dialect/ring/transforms/pass_details.h"
#include "libspu/dialect/utils/lowering_intrinsic.h"
#include "libspu/dialect/utils/utils.h"

namespace mlir::spu {
namespace {

Type stripSecretType(Type in) {
  if (auto rt = mlir::dyn_cast<RankedTensorType>(in)) {
    return RankedTensorType::get(rt.getShape(),
                                 stripSecretType(rt.getElementType()));
  }

  if (auto st = mlir::dyn_cast<ring::SecretType>(in)) {
    return st.getBaseType();
  }

  return in;
}

Type buildRingType(MLIRContext *context, int64_t width,
                   bool is_unsigned = false) {
  return IntegerType::get(context, width,
                          is_unsigned
                              ? IntegerType::SignednessSemantics::Unsigned
                              : IntegerType::SignednessSemantics::Signless);
}

Type replaceRingType(Type in, Type new_ring) {
  if (auto rt = mlir::dyn_cast<RankedTensorType>(in)) {
    return rt.clone(replaceRingType(rt.getElementType(), new_ring));
  }
  if (auto st = mlir::dyn_cast<ring::SecretType>(in)) {
    return ring::SecretType::get(new_ring);
  }
  return new_ring;
}

template <typename SHIFT_T>
Value buildShift(OpBuilder &builder, Value lhs, int8_t bits) {
  auto const_t = cast<ShapedType>(stripSecretType(lhs.getType()));

  APInt bits_(const_t.getElementType().getIntOrFloatBitWidth(), bits);

  auto bits_v = builder.create<arith::ConstantOp>(
      lhs.getLoc(), DenseIntElementsAttr::get(const_t, bits_));

  return builder.create<SHIFT_T>(lhs.getLoc(), lhs.getType(), lhs, bits_v);
}

template <typename OP>
class RingOpUnaryConverter : public OpConversionPattern<OP> {
 private:
  pphlo::TypeTools tools_;

 public:
  RingOpUnaryConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<OP>(type_converter, context), tools_(context) {}

  LogicalResult matchAndRewrite(
      OP op, typename RingOpUnaryConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type result_type =
        this->typeConverter->convertType(op->getResultTypes()[0]);

    auto new_op = rewriter.create<PPHloToRingOpDirect<OP>>(
        op->getLoc(), result_type, adaptor.getOperand());

    new_op->setAttrs(op->getAttrs());

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <typename OP>
class RingOpBinaryConverter : public OpConversionPattern<OP> {
 private:
  pphlo::TypeTools tools_;

 public:
  RingOpBinaryConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<OP>(type_converter, context), tools_(context) {}

  LogicalResult matchAndRewrite(
      OP op, typename RingOpBinaryConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumResults() == 1);

    Type result_type =
        this->typeConverter->convertType(op->getResultTypes()[0]);

    auto new_op = rewriter.create<PPHloToRingOpDirect<OP>>(
        op->getLoc(), result_type, adaptor.getLhs(), adaptor.getRhs());

    new_op->setAttrs(op->getAttrs());

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
class RingOpUnaryConverter<pphlo::BitcastConvertOp>
    : public OpConversionPattern<pphlo::BitcastConvertOp> {
 public:
  RingOpUnaryConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<pphlo::BitcastConvertOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      pphlo::BitcastConvertOp op, pphlo::BitcastConvertOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type result_type =
        this->typeConverter->convertType(op->getResultTypes()[0]);

    pphlo::TypeTools tools(op->getContext());

    Value new_op;
    if (tools.isPublicType(op.getType())) {
      new_op = rewriter.create<tensor::BitcastOp>(op->getLoc(), result_type,
                                                  adaptor.getOperand());
    } else {
      new_op = rewriter.create<ring::CastOp>(op->getLoc(), result_type,
                                             adaptor.getOperand());
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
class RingOpUnaryConverter<pphlo::TruncOp>
    : public OpConversionPattern<pphlo::TruncOp> {
 private:
  pphlo::TypeTools tools_;

 public:
  RingOpUnaryConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<pphlo::TruncOp>(type_converter, context),
        tools_(context) {}

  LogicalResult matchAndRewrite(
      pphlo::TruncOp op, pphlo::TruncOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto in_fxp_bits = tools_.getFxpBits(op.getOperand().getType());
    auto out_fxp_bits = tools_.getFxpBits(op.getResult().getType());

    if (out_fxp_bits > in_fxp_bits) {
      return op.emitOpError("Truncation bits should not be negative");
    }

    rewriter.replaceOpWithNewOp<ring::TruncOp>(
        op, adaptor.getOperand().getType(), adaptor.getOperand(),
        in_fxp_bits - out_fxp_bits);

    return success();
  }
};

template <>
class RingOpUnaryConverter<pphlo::ConvertOp>
    : public OpConversionPattern<pphlo::ConvertOp> {
 private:
  pphlo::TypeTools tools_;

 public:
  RingOpUnaryConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<pphlo::ConvertOp>(type_converter, context),
        tools_(context) {}

  LogicalResult matchAndRewrite(
      pphlo::ConvertOp op, pphlo::ConvertOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = typeConverter->convertType(op.getResult().getType());

    Value result = adaptor.getOperand();
    // v cast?
    auto in_vis = tools_.getTypeVisibility(op.getOperand().getType());
    auto out_vis = tools_.getTypeVisibility(op.getResult().getType());

    auto in_el_type = tools_.getBaseType(op.getOperand().getType());
    auto out_el_type = tools_.getBaseType(op.getResult().getType());

    if (in_vis != out_vis) {
      if (in_el_type != out_el_type) {
        return emitOptionalError(
            op->getLoc(),
            "Should not do vcast and real type cast in one cast op");
      }
      // If this conversion has s2p, make 2p first
      if (in_vis == pphlo::Visibility::SECRET &&
          out_vis == pphlo::Visibility::PUBLIC) {
        // s2p
        return emitOptionalError(op->getLoc(),
                                 "Should not create S2P from ConverterOp");
      }
      if (in_vis == pphlo::Visibility::PUBLIC &&
          out_vis == pphlo::Visibility::SECRET) {
        // p2s
        rewriter.replaceOpWithNewOp<ring::P2SOp>(op, result_type,
                                                 adaptor.getOperand());
        return success();
      }
    }

    if (in_vis == pphlo::Visibility::PUBLIC && in_vis == out_vis) {
      return emitOptionalError(
          op->getLoc(), "Pure public conversion should already been handled");
    }

    int64_t out_width = 0;
    int64_t in_fxp_bits = 0;
    int64_t out_fxp_bits = 0;
    bool out_is_unsigned = false;

    if (auto in_t = mlir::dyn_cast<pphlo::FixedPointType>(in_el_type)) {
      in_fxp_bits = in_t.getFraction();
    }

    if (auto out_t = mlir::dyn_cast<IntegerType>(out_el_type)) {
      out_width = out_t.getWidth();
      out_is_unsigned = out_t.isUnsigned();
    } else if (auto out_t =
                   mlir::dyn_cast<pphlo::FixedPointType>(out_el_type)) {
      out_width = out_t.getWidth();
      out_fxp_bits = out_t.getFraction();
    }

    auto result_ring = replaceRingType(
        result.getType(),
        buildRingType(op->getContext(), out_width, out_is_unsigned));
    if (out_fxp_bits == 0 && in_fxp_bits > 0) {
      // int2fxp
      // (x + 0.99 * (x < 0)) >> fxp_bits
      auto msb = rewriter.create<ring::MsbOp>(op->getLoc(), result);
      auto const_type = cast<ShapedType>(stripSecretType(result.getType()));
      auto oneMinusEps = rewriter.create<arith::ConstantOp>(
          op->getLoc(),
          DenseIntElementsAttr::get(
              const_type,
              APInt(const_type.getElementType().getIntOrFloatBitWidth(),
                    (static_cast<uint64_t>(1) << in_fxp_bits) - 1)));
      auto mul = rewriter.create<ring::MulOp>(op->getLoc(), result.getType(),
                                              oneMinusEps, msb);
      result = rewriter.create<ring::AddOp>(op->getLoc(), result, mul);
      result = buildShift<ring::ARShiftOp>(rewriter, result, in_fxp_bits);
      // ring cast
      result = rewriter.create<ring::CastOp>(op->getLoc(), result_ring, result);
    } else if (out_fxp_bits < in_fxp_bits) {
      // Cast to a type with less fxp bits, shift first and shrink ring after
      result = buildShift<ring::ARShiftOp>(rewriter, result,
                                           in_fxp_bits - out_fxp_bits);
      result = rewriter.create<ring::CastOp>(op->getLoc(), result_ring, result);
    } else if (out_fxp_bits > in_fxp_bits) {
      // Cast to a type with more fxp bits, increate ring first and lshift
      // after
      result = rewriter.create<ring::CastOp>(op->getLoc(), result_ring, result);
      result = buildShift<ring::LShiftOp>(rewriter, result,
                                          out_fxp_bits - in_fxp_bits);
    } else {
      result = rewriter.create<ring::CastOp>(op->getLoc(), result_ring, result);
    }

    rewriter.replaceOp(op, result);

    return success();
  }
};

template <>
class RingOpUnaryConverter<pphlo::BitRevOp>
    : public OpConversionPattern<pphlo::BitRevOp> {
 public:
  RingOpUnaryConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<pphlo::BitRevOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      pphlo::BitRevOp op, pphlo::BitRevOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ret_type = typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<ring::BitRevOp>(
        op, ret_type, adaptor.getOperand(), op.getStart(), op.getEnd());

    return success();
  }
};

class DynamicUpdateSliceOpConverter
    : public OpConversionPattern<pphlo::DynamicUpdateSliceOp> {
 public:
  DynamicUpdateSliceOpConverter(TypeConverter &type_converter,
                                MLIRContext *context)
      : OpConversionPattern<pphlo::DynamicUpdateSliceOp>(type_converter,
                                                         context) {}

  LogicalResult matchAndRewrite(
      pphlo::DynamicUpdateSliceOp op,
      pphlo::DynamicUpdateSliceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result_type = mlir::dyn_cast<ShapedType>(
        this->typeConverter->convertType(op->getResultTypes()[0]));

    llvm::SmallVector<Value> scalar_start_indices(op.getStartIndices().size());

    for (size_t idx = 0; idx < scalar_start_indices.size(); ++idx) {
      scalar_start_indices[idx] = rewriter.create<tensor::ExtractOp>(
          op->getLoc(), adaptor.getStartIndices()[idx]);
    }

    rewriter.replaceOpWithNewOp<ring::SecretInsertSliceOp>(
        op, result_type, adaptor.getOperand(), adaptor.getUpdate(),
        scalar_start_indices);

    return success();
  }
};

class CustomCallOpConverter : public OpConversionPattern<pphlo::CustomCallOp> {
 private:
  std::string mangle_name(llvm::StringRef in, TypeRange in_types,
                          TypeRange out_types) const {
    auto ret = in.str();

    if (in_types.empty() && out_types.empty()) {
      return ret;
    }

    ret += "#";

    for (auto types : {in_types, out_types}) {
      for (auto t : types) {
        auto el = getElementTypeOrSelf(t);
        if (auto st = mlir::dyn_cast<ring::SecretType>(el)) {
          ret += "s";
          el = st.getBaseType();
        }
        ret += fmt::format(
            "{}", fmt::join(mlir::cast<ShapedType>(t).getShape(), "x"));
        ret += mlirObjectToString(el);
        ret += "_";
      }
    }

    return ret;
  }

  SmallVector<NamedAttribute> pruneAttributes(
      ArrayRef<NamedAttribute> attrs) const {
    SmallVector<NamedAttribute> results;

    for (const auto &attr : attrs) {
      if (attr.getName() == "has_side_effect") {
        continue;
      }
      if (attr.getName() == "allow_float") {
        continue;
      }
      if (attr.getName() == "call_target_name") {
        continue;
      }
      results.emplace_back(attr);
    }

    return results;
  }

  LogicalResult rewriterToFuncCall(pphlo::CustomCallOp op,
                                   pphlo::CustomCallOpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    llvm::SmallVector<Type> result_types;

    if (typeConverter->convertTypes(op->getResultTypes(), result_types)
            .failed()) {
      return failure();
    }

    auto *module = SymbolTable::getNearestSymbolTable(op);

    // Expected signature
    auto in_types = llvm::to_vector(adaptor.getInputs().getType());

    auto mangled_name =
        mangle_name(op.getCallTargetName(), in_types, result_types);

    // replace with func::CallOp
    auto opFunc = mlir::dyn_cast_or_null<mlir::SymbolOpInterface>(
        mlir::SymbolTable::lookupSymbolIn(module, mangled_name));
    if (opFunc == nullptr) {
      OpBuilder::InsertionGuard guard(rewriter);

      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      auto fcnType =
          FunctionType::get(rewriter.getContext(), in_types, result_types);
      opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(),
                                             mangled_name, fcnType);
      opFunc.setPrivate();
    }

    auto callop = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, mangled_name, result_types, adaptor.getOperands());

    callop->setAttrs(pruneAttributes(op->getAttrs()));

    return success();
  }

 public:
  CustomCallOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<pphlo::CustomCallOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      pphlo::CustomCallOp op, pphlo::CustomCallOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> result_types;

    if (typeConverter->convertTypes(op->getResultTypes(), result_types)
            .failed()) {
      return failure();
    }

    if (op.getCallTargetName() == ENCODE_TO_FXP) {
      pphlo::TypeTools tools(op->getContext());
      rewriter.replaceOpWithNewOp<ring::EncodeToFxpOp>(
          op, result_types.front(), adaptor.getOperands()[0],
          tools.getFxpBits(op->getResultTypes()[0]));
    } else if (op.getCallTargetName() == DECODE_FROM_FXP) {
      pphlo::TypeTools tools(op->getContext());
      rewriter.replaceOpWithNewOp<ring::DecodeFromFxpOp>(
          op, result_types.front(), adaptor.getOperands()[0],
          tools.getFxpBits(op->getOperandTypes()[0]));
    } else if (op.getCallTargetName() == SECRET_INDEX) {
      auto indexing_dim = llvm::map_to_vector(
          mlir::cast<mlir::ArrayAttr>(op->getAttr("indexing_dim")),
          [](Attribute attr) {
            return mlir::cast<IntegerAttr>(attr).getInt();
          });
      rewriter.replaceOpWithNewOp<ring::SecretExtractSliceOp>(
          op, result_types.front(), adaptor.getOperands().back(),
          ValueRange{llvm::drop_end(adaptor.getOperands(), 1)}, indexing_dim);
    } else if (op.getCallTargetName() == PARTIAL_MUL) {
      rewriter.replaceOpWithNewOp<ring::MulOp>(op, result_types[0],
                                               adaptor.getOperands()[0],
                                               adaptor.getOperands()[1]);
    } else if (op.getCallTargetName() == PARTIAL_DOT) {
      rewriter.replaceOpWithNewOp<ring::DotOp>(op, result_types[0],
                                               adaptor.getOperands()[0],
                                               adaptor.getOperands()[1]);
    } else {
      return rewriterToFuncCall(op, adaptor, rewriter);
    }
    return success();
  }
};

Value extractTensorValue(OpBuilder &b, Value tensor) {
  auto loc = tensor.getLoc();
  if (mlir::cast<TensorType>(tensor.getType()).hasRank() &&
      mlir::cast<TensorType>(tensor.getType()).getRank() != 0) {
    tensor = b.create<tensor::CollapseShapeOp>(
        loc, tensor, SmallVector<ReassociationIndices>());
  }
  return b.create<tensor::ExtractOp>(loc, tensor, ValueRange());
}

template <typename OP>
class RingOpShiftConverter : public OpConversionPattern<OP> {
 private:
  pphlo::TypeTools tools_;

 public:
  RingOpShiftConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<OP>(type_converter, context), tools_(context) {}

  LogicalResult matchAndRewrite(
      OP op, typename RingOpShiftConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type result_type =
        this->typeConverter->convertType(op->getResultTypes()[0]);

    if constexpr (std::is_same_v<OP, pphlo::ShiftLeftOp>) {
      rewriter.replaceOpWithNewOp<ring::LShiftOp>(
          op, result_type, adaptor.getLhs(), adaptor.getRhs());
      return success();
    } else if constexpr (std::is_same_v<OP, pphlo::ShiftRightLogicalOp>) {
      rewriter.replaceOpWithNewOp<ring::RShiftOP>(
          op, result_type, adaptor.getLhs(), adaptor.getRhs());
      return success();
    } else if constexpr (std::is_same_v<OP, pphlo::ShiftRightArithmeticOp>) {
      rewriter.replaceOpWithNewOp<ring::ARShiftOp>(
          op, result_type, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
  }
};

class ReturnOpConverter : public OpConversionPattern<func::ReturnOp> {
 public:
  ReturnOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<mlir::func::ReturnOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      ::mlir::func::ReturnOp op, ::mlir::func::ReturnOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    rewriter.modifyOpInPlace(
        op, [&]() { operation->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class PPHloReturnOpConverter : public OpConversionPattern<pphlo::ReturnOp> {
 private:
  void ensurePrototypeExist(pphlo::ReturnOp op, OpBuilder &builder,
                            Type result_type, Type in_type) const {
    auto *module = mlir::SymbolTable::getNearestSymbolTable(op);

    if (mlir::SymbolTable::lookupSymbolIn(module, TRY_REVEAL_COND) == nullptr) {
      OpBuilder::InsertionGuard guard(builder);

      builder.setInsertionPointToStart(&module->getRegion(0).front());

      auto fcnType = FunctionType::get(builder.getContext(), TypeRange{in_type},
                                       TypeRange{result_type});
      auto fcn = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                              TRY_REVEAL_COND, fcnType);
      fcn.setPrivate();
    }
  }

  Value extractWhileConditionTensorValue(pphlo::ReturnOp op, OpBuilder &b,
                                         Value tensor) const {
    auto loc = tensor.getLoc();
    auto scalar = extractTensorValue(b, tensor);

    if (auto st = mlir::dyn_cast<ring::SecretType>(scalar.getType())) {
      ensurePrototypeExist(op, b, st.getBaseType(), scalar.getType());
      return b
          .create<func::CallOp>(loc, TRY_REVEAL_COND,
                                TypeRange{st.getBaseType()}, ValueRange{scalar})
          ->getResult(0);
    }
    return scalar;
  }

 public:
  PPHloReturnOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<pphlo::ReturnOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      pphlo::ReturnOp op, pphlo::ReturnOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // linalg reduce need a linalg.yield as terminator
    if (auto linalg_reduce =
            dyn_cast<linalg::ReduceOp>(op->getParentRegion()->getParentOp())) {
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

    return failure();
  }
};

template <typename T>
struct StandardTypeOnlyConverter : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      T op, typename StandardTypeOnlyConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Converter result types
    llvm::SmallVector<Type> result_types;
    if (failed(this->typeConverter->convertTypes(op->getResultTypes(),
                                                 result_types))) {
      return failure();
    }

    // create new op
    Operation *new_op = nullptr;

    if constexpr (std::is_same_v<T, scf::IfOp>) {
      new_op = rewriter.create<T>(op.getLoc(), result_types,
                                  adaptor.getCondition(), true, true);
    } else {
      new_op = rewriter.create<T>(op.getLoc(), result_types,
                                  adaptor.getOperands(), op->getAttrs());
    }

    // Convert regions
    for (size_t idx = 0; idx < op->getNumRegions(); ++idx) {
      auto &source_region = op->getRegions()[idx];
      auto &dest_region = new_op->getRegions()[idx];

      TypeConverter::SignatureConversion sig_conversion(
          source_region.getNumArguments());

      for (const auto &arg : source_region.getArguments()) {
        auto arg_t = this->typeConverter->convertType(arg.getType());
        sig_conversion.addInputs(arg.getArgNumber(), arg_t);
      }

      // inline region
      rewriter.inlineRegionBefore(source_region, dest_region,
                                  dest_region.end());

      // Convert region
      if (failed(rewriter.convertRegionTypes(
              &dest_region, *this->getTypeConverter(), &sig_conversion))) {
        return op->emitOpError("Failed to convert region");
      }
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<linalg::BroadcastOp>
    : public OpConversionPattern<linalg::BroadcastOp> {
  using OpConversionPattern<linalg::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::BroadcastOp op, linalg::BroadcastOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
        op, adaptor.getInput(), adaptor.getInit(), op.getDimensions());
    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<linalg::TransposeOp>
    : public OpConversionPattern<linalg::TransposeOp> {
  using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::TransposeOp op, linalg::TransposeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getInput(), adaptor.getInit(), op.getPermutation());
    return success();
  }
};

class FuncOpConverter : public OpConversionPattern<::mlir::func::FuncOp> {
 public:
  FuncOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<::mlir::func::FuncOp>(type_converter, context) {}

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
        auto convertedType =
            typeConverter->convertType(blockArgument.getType());

        conversion.addInputs(idx, convertedType);
      }
      rewriter.applySignatureConversion(&block, conversion, getTypeConverter());
    }

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(functionType.getNumInputs());
    for (const auto &blockarg : llvm::enumerate(op.getBody().getArguments())) {
      auto convertedType =
          typeConverter->convertType(blockarg.value().getType());
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
    op.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                        newResultTypes));
    rewriter.finalizeOpModification(op);

    return success();
  }
};

class PPHLOToRingTypeConverter : public TypeConverter {
 private:
  static Type convertIntegerType(IntegerType type) {
    return buildRingType(type.getContext(), type.getWidth(), type.isUnsigned());
  }

  static Type convertFxpType(pphlo::FixedPointType type) {
    return buildRingType(type.getContext(), type.getWidth(), false);
  }

  static Type convertSecretType(pphlo::SecretType type) {
    Type base;
    if (auto it = mlir::dyn_cast<IntegerType>(type.getBaseType())) {
      base = convertIntegerType(it);
    }
    if (auto ft = mlir::dyn_cast<pphlo::FixedPointType>(type.getBaseType())) {
      base = convertFxpType(ft);
    }

    return ring::SecretType::get(base);
  }

  static std::optional<Value> scalarToTensor(OpBuilder &builder, Type type,
                                             ValueRange inputs, Location loc) {
    assert(inputs.size() == 1);
    if (llvm::isa<ShapedType>(inputs.front().getType())) {
      return std::nullopt;
    }

    Value result =
        builder
            .create<tensor::FromElementsOp>(
                loc, RankedTensorType::get({}, inputs.front().getType()),
                inputs.front())
            .getResult();

    Type elementType = mlir::getElementTypeOrSelf(type);
    if (inputs.front().getType() != elementType) {
      result = builder.create<UnrealizedConversionCastOp>(loc, type, result)
                   ->getResult(0);
    }

    return result;
  }

  static std::optional<Value> materializeCastFromIllegal(OpBuilder &builder,
                                                         Type type,
                                                         ValueRange inputs,
                                                         Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
  }

 public:
  explicit PPHLOToRingTypeConverter() {
    addConversion([](IndexType t) { return t; });
    addConversion([](FloatType t) { return t; });
    addConversion([](ring::SecretType t) { return t; });
    addConversion(
        [&](IntegerType type) -> Type { return convertIntegerType(type); });
    addConversion([&](pphlo::FixedPointType type) -> Type {
      return convertFxpType(type);
    });
    addConversion([&](pphlo::SecretType t) { return convertSecretType(t); });
    addConversion([&](ShapedType type) -> Type {
      auto eltype = type.getElementType();
      if (auto ft = mlir::dyn_cast<IntegerType>(eltype)) {
        return type.clone(convertIntegerType(ft));
      }
      if (auto fxpt = mlir::dyn_cast<pphlo::FixedPointType>(eltype)) {
        return type.clone(convertFxpType(fxpt));
      }
      if (auto st = mlir::dyn_cast<pphlo::SecretType>(eltype)) {
        return type.clone(convertSecretType(st));
      }
      return type;
    });
    addArgumentMaterialization(scalarToTensor);
    addTargetMaterialization(materializeCastFromIllegal);
  }
};

struct LegalizeToRing : public ring::PPHloLegalizeToRingBase<LegalizeToRing> {
 private:
  template <typename OpT>
  void addTypeOnlyConversion(RewritePatternSet &patterns,
                             PPHLOToRingTypeConverter &converter,
                             MLIRContext *context) {
    patterns.insert<StandardTypeOnlyConverter<OpT>>(converter, context);
  }

  template <typename OpT, typename OpT2, typename... OpTs>
  void addTypeOnlyConversion(RewritePatternSet &patterns,
                             PPHLOToRingTypeConverter &converter,
                             MLIRContext *context) {
    addTypeOnlyConversion<OpT>(patterns, converter, context);
    addTypeOnlyConversion<OpT2, OpTs...>(patterns, converter, context);
  }

  void populateConversionPattern(PPHLOToRingTypeConverter &converter,
                                 RewritePatternSet &patterns) {
    auto *context = patterns.getContext();

    patterns.insert<FuncOpConverter>(converter, context);

    // Unary
    patterns.insert<RingOpUnaryConverter<pphlo::TruncOp>,           //
                    RingOpUnaryConverter<pphlo::ConvertOp>,         //
                    RingOpUnaryConverter<pphlo::BitcastConvertOp>,  //
                    RingOpUnaryConverter<pphlo::BitRevOp>,          //
                    RingOpUnaryConverter<pphlo::NegOp>,             //
                    RingOpUnaryConverter<pphlo::NotOp>>(converter, context);

    // Binary
    patterns.insert<RingOpBinaryConverter<pphlo::AddOp>,                  //
                    RingOpBinaryConverter<pphlo::AndOp>,                  //
                    RingOpBinaryConverter<pphlo::EqualOp>,                //
                    RingOpBinaryConverter<pphlo::MulOp>,                  //
                    RingOpBinaryConverter<pphlo::XorOp>,                  //
                    RingOpBinaryConverter<pphlo::DotOp>,                  //
                    RingOpShiftConverter<pphlo::ShiftLeftOp>,             //
                    RingOpShiftConverter<pphlo::ShiftRightLogicalOp>,     //
                    RingOpShiftConverter<pphlo::ShiftRightArithmeticOp>,  //
                    RingOpBinaryConverter<pphlo::LessOp>>(converter, context);

    // ShapeOps
    patterns.insert<DynamicUpdateSliceOpConverter>(converter, context);

    // Controlflow
    patterns.insert<PPHloReturnOpConverter>(converter, context);

    // Others
    patterns.insert<ReturnOpConverter, CustomCallOpConverter>(converter,
                                                              context);

    addTypeOnlyConversion<
#include "libspu/dialect/ring/transforms/non_ring_op_list.h.inc"
        >(patterns, converter, context);
  }

 public:
  LegalizeToRing(const LegalizeToRing &) = default;
  LegalizeToRing() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    PPHLOToRingTypeConverter converter;

    target.addLegalDialect<ring::RingDialect, mlir::tensor::TensorDialect,
                           mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                           mlir::math::MathDialect,
                           mlir::linalg::LinalgDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<pphlo::PPHloDialect>();

    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });

    target.addDynamicallyLegalOp<  //
        func::ReturnOp,            //
        func::CallOp,              //
#include "libspu/dialect/ring/transforms/non_ring_op_list.h.inc"
        >([&](Operation *op) { return converter.isLegal(op); });

    populateConversionPattern(converter, patterns);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

namespace ring {

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToRing() {
  return std::make_unique<LegalizeToRing>();
}

}  // namespace ring
}  // namespace mlir::spu
