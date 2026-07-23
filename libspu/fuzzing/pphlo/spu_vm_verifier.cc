// Copyright 2023 Ant Group Co., Ltd.
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

// TODO: generate pphlo.unset<i32>
// TODO: control flow op

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

// MLIR Generation Headers
#include <mlir/IR/Dialect.h>

#include <type_traits>

#include "FuzzedDataProvider.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "libspu/device/utils/pphlo_executor_test_runner.h"
#include "libspu/dialect/pphlo/IR/dialect.h"
#include "libspu/dialect/pphlo/IR/ops.h"

#include "libspu/spu.pb.h"

namespace dx::operator_utils {

template <typename T>
constexpr bool isOneOfType() {
  return false;
}

template <typename T, typename U, typename... Args>
constexpr bool isOneOfType() {
  if constexpr (std::is_same_v<T, U>) {
    return true;
  }
  return isOneOfType<T, Args...>();
}

}  // namespace dx::operator_utils

static std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::spu::pphlo::PPHloDialect, mlir::BuiltinDialect,
                  mlir::func::FuncDialect>();
  std::unique_ptr<mlir::MLIRContext> ctx =
      std::make_unique<mlir::MLIRContext>(registry);
  ctx->getOrLoadDialect<mlir::spu::pphlo::PPHloDialect>();
  ctx->getOrLoadDialect<mlir::BuiltinDialect>();
  ctx->getOrLoadDialect<mlir::func::FuncDialect>();
  return ctx;
}

template <typename OpTy>
constexpr bool isIntTensorOp() {  // UnaryElementwiseOps accept int tensor.
  return dx::operator_utils::isOneOfType<OpTy, mlir::spu::pphlo::NotOp>();
}

template <typename OpTy>
constexpr bool isFloatTensorOp() {  // UnaryElementwiseOps accept float tensor.
  return dx::operator_utils::isOneOfType<
      OpTy, mlir::spu::pphlo::SqrtOp, mlir::spu::pphlo::RsqrtOp,
      mlir::spu::pphlo::TanhOp, mlir::spu::pphlo::ExpOp,
      mlir::spu::pphlo::Expm1Op, mlir::spu::pphlo::LogOp,
      mlir::spu::pphlo::Log1pOp, mlir::spu::pphlo::CeilOp,
      mlir::spu::pphlo::FloorOp, mlir::spu::pphlo::RoundOp,
      mlir::spu::pphlo::ReciprocalOp, mlir::spu::pphlo::LogisticOp>();
}

template <typename OpTy>
constexpr bool
isAnyTensorOp() {  // UnaryElementwiseOps accept both int and float tensor.
  return dx::operator_utils::isOneOfType<OpTy, mlir::spu::pphlo::NegOp,
                                         mlir::spu::pphlo::AbsOp,
                                         mlir::spu::pphlo::SignOp>();
}

template <typename T, uint64_t pos>
void compile_time_assertion_failure_impl(const char *fmt);

#define compile_time_assertion_failure(T, fmt)             \
  do {                                                     \
    compile_time_assertion_failure_impl<T, __LINE__>(fmt); \
  } while (0)

struct Mutator {
  explicit Mutator(mlir::MLIRContext &ctx) : ctx(ctx), builder(&ctx) {}

  // Generate random shape.
  static spu::Shape genShape(FuzzedDataProvider &fuzzed_data) {
    auto rank = fuzzed_data.ConsumeIntegralInRange<uint8_t>(
        0, 5);  // TODO: more intelligent rank value
    spu::Shape shape(rank);
    for (int i = 0; i < rank; i++) {
      shape[i] = fuzzed_data.ConsumeIntegralInRange<uint8_t>(
          1,
          10);  // TODO: more intelligent elements number value
    }
    return shape;
  }

  // Generate random int/float element type
  std::pair<mlir::Type, bool> genElementType(FuzzedDataProvider &fuzzed_data) {
    mlir::Type element_type;
    bool is_float = false;
    if (fuzzed_data.ConsumeBool()) {  // Return Integer Type
      is_float = false;
      switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 3)) {
        case 0:
          element_type = builder.getIntegerType(8);
          break;
        case 1:
          element_type = builder.getIntegerType(16);
          break;
        case 2:
          element_type = builder.getIntegerType(32);
          break;
        case 3:
          element_type = builder.getIntegerType(64);
          break;
        default:
          abort();
      }

    } else {  // Return Float Type
      is_float = true;
      switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 1)) {
        case 0:
          element_type = builder.getF32Type();
          break;
        case 1:
          element_type = builder.getF64Type();
          break;
        default:
          abort();
      }
    }
    std::pair<mlir::Type, bool> result = std::make_pair(element_type, is_float);
    return result;
  }

  // Generate random float element type.
  mlir::Type genFPElementType(FuzzedDataProvider &fuzzed_data) {
    mlir::Type element_type;
    switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 1)) {
      case 0:
        element_type = builder.getF32Type();
        break;
      case 1:
        element_type = builder.getF64Type();
        break;
      default:
        abort();
    }
    return element_type;
  }

  // Generate random int element type.
  mlir::Type genIntElementType(FuzzedDataProvider &fuzzed_data) {
    mlir::Type element_type;
    switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 3)) {
      case 0:
        element_type = builder.getIntegerType(8);
        break;
      case 1:
        element_type = builder.getIntegerType(16);
        break;
      case 2:
        element_type = builder.getIntegerType(32);
        break;
      case 3:
        element_type = builder.getIntegerType(64);
        break;
      default:
        abort();
    }

    return element_type;
  }

  // Generate shaped pphlo public/secret tensor type
  std::pair<mlir::RankedTensorType, bool> genShapedPPHloTensorType(
      FuzzedDataProvider &fuzzed_data) {
    mlir::Type shapedPPHloPublicTensorType;
    std::pair<mlir::Type, bool> element_type_res = genElementType(fuzzed_data);
    mlir::Type element_type = element_type_res.first;
    bool is_float = element_type_res.second;
    std::vector<int64_t> shape = genShape(fuzzed_data);
    mlir::spu::pphlo::TypeTools typetools(&ctx);
    if (fuzzed_data.ConsumeBool()) {  // return public type
      shapedPPHloPublicTensorType =
          typetools.getType(mlir::RankedTensorType::get(shape, element_type),
                            mlir::spu::pphlo::Visibility::PUBLIC);
    } else {  // return secret type
      shapedPPHloPublicTensorType =
          typetools.getType(mlir::RankedTensorType::get(shape, element_type),
                            mlir::spu::pphlo::Visibility::SECRET);
    }
    std::pair<mlir::RankedTensorType, bool> result = std::make_pair(
        mlir::dyn_cast<mlir::RankedTensorType>(shapedPPHloPublicTensorType),
        is_float);
    return result;
  }

  // Generate specific bit_width DenseIntElementsAttr.
  template <int bit_width, typename elem_ty>
  mlir::DenseIntElementsAttr genDenseInt(FuzzedDataProvider &fuzzed_data) {
    if constexpr (bit_width != 8 && bit_width != 16 && bit_width != 32 &&
                  bit_width != 64) {
      compile_time_assertion_failure(elem_ty, "invalid bit width");
    }
    auto int_ty = builder.getIntegerType(bit_width);
    auto shape = genShape(fuzzed_data);
    int64_t elements_num = shape.numel();
    std::vector<elem_ty> int_values(elements_num);
    for (int i = 0; i < elements_num; i++) {
      int_values[i] = fuzzed_data.ConsumeIntegral<elem_ty>();
    }
    mlir::RankedTensorType ranked_tensor_type =
        mlir::RankedTensorType::get(shape, int_ty);
    return mlir::DenseIntElementsAttr::get(ranked_tensor_type, int_values);
  }

  // Generate specific bit_width DenseIntElementsAttr with specific shape.
  template <int bit_width, typename elem_ty>
  mlir::DenseIntElementsAttr genDenseInt(FuzzedDataProvider &fuzzed_data,
                                         const spu::Shape &shape) {
    if constexpr (bit_width != 8 && bit_width != 16 && bit_width != 32 &&
                  bit_width != 64) {
      compile_time_assertion_failure(elem_ty, "invalid bit width");
    }
    auto int_ty = builder.getIntegerType(bit_width);
    int64_t elements_num = shape.numel();
    std::vector<elem_ty> int_values(elements_num);
    for (int i = 0; i < elements_num; i++) {
      int_values[i] = fuzzed_data.ConsumeIntegral<elem_ty>();
    }
    mlir::RankedTensorType ranked_tensor_type =
        mlir::RankedTensorType::get(shape, int_ty);
    return mlir::DenseIntElementsAttr::get(ranked_tensor_type, int_values);
  }

  // Generate specific bit_width DenseFPElementsAttr.
  template <int bit_width, typename elem_ty>
  mlir::DenseFPElementsAttr genDenseFloat(FuzzedDataProvider &fuzzed_data) {
    mlir::Type flt_ty;
    if constexpr (bit_width == 32) {
      flt_ty = builder.getF32Type();
    } else if constexpr (bit_width == 64) {
      flt_ty = builder.getF64Type();
    } else {
      compile_time_assertion_failure(elem_ty, "invalid bit width");
    }
    auto shape = genShape(fuzzed_data);
    int64_t elements_num = shape.numel();
    std::vector<elem_ty> flt_values(elements_num);
    for (int i = 0; i < elements_num; i++) {
      flt_values[i] = fuzzed_data.ConsumeFloatingPoint<elem_ty>();
    }
    mlir::RankedTensorType ranked_tensor_type =
        mlir::RankedTensorType::get(shape, flt_ty);
    return mlir::DenseFPElementsAttr::get(ranked_tensor_type, flt_values);
  }

  // Generate specific bit_width DenseFPElementsAttr with specific shape.
  template <int bit_width, typename elem_ty>
  mlir::DenseFPElementsAttr genDenseFloat(FuzzedDataProvider &fuzzed_data,
                                          const spu::Shape &shape) {
    mlir::Type flt_ty;
    if constexpr (bit_width == 32) {
      flt_ty = builder.getF32Type();
    } else if constexpr (bit_width == 64) {
      flt_ty = builder.getF64Type();
    } else {
      compile_time_assertion_failure(elem_ty, "invalid bit width");
    }
    int64_t elements_num = shape.numel();

    std::vector<elem_ty> flt_values(elements_num);
    for (int i = 0; i < elements_num; i++) {
      flt_values[i] = fuzzed_data.ConsumeFloatingPoint<elem_ty>();
    }
    mlir::RankedTensorType ranked_tensor_type =
        mlir::RankedTensorType::get(shape, flt_ty);
    return mlir::DenseFPElementsAttr::get(ranked_tensor_type, flt_values);
  }

  template <typename OpTy>
  void addDimTensor(OpTy op) {
    auto ty = op.getType();
    auto dim_size = ty.getShape().size();
    if (dim_size >= 10) {
      return;
    }
    dim_tensor_values[dim_size].push_back(op);
  }

  template <typename OpTy>
  OpTy addIntTensor(OpTy op) {  // add op into the int tensors list
    addDimTensor(op);
    int_tensor_values.push_back(op);
    return op;
  }

  template <typename OpTy>
  OpTy addFltTensor(OpTy op) {  // add op into the float tensors list
    addDimTensor(op);
    flt_tensor_values.push_back(op);
    return op;
  }

  // Create a new int tensor or return an int tensor from created int tensors.
  mlir::Value sampleIntTensorOp(FuzzedDataProvider &fuzzed_data) {
    if (int_tensor_values.empty()) {
      return createConstantIntOp(fuzzed_data);
    } else {
      auto probability = fuzzed_data.ConsumeProbability<float>();
      // 70% probability creates new tensor op, 30% probability uses exist
      // tensor op.
      if (probability < 0.7) {
        return int_tensor_values[fuzzed_data.ConsumeIntegralInRange<uint16_t>(
            0, int_tensor_values.size() - 1)];
      } else {
        return createConstantIntOp(fuzzed_data);
      }
    }
  }

  // Create a new float tensor or return a float tensor from created float
  // tensors.
  mlir::Value sampleFltTensorOp(FuzzedDataProvider &fuzzed_data) {
    if (flt_tensor_values.empty()) {
      return createConstantFloatOp(fuzzed_data);
    } else {
      auto probability = fuzzed_data.ConsumeProbability<float>();
      // 70% probability creates new tensor op, 30% probability uses exist
      // tensor op.
      if (probability < 0.7) {
        return flt_tensor_values[fuzzed_data.ConsumeIntegralInRange<uint16_t>(
            0, flt_tensor_values.size() - 1)];
      } else {
        return createConstantFloatOp(fuzzed_data);
      }
    }
  }

  // Return a float or an int tensor.
  std::tuple<mlir::Value, bool> sampleAnyTensorOp(
      FuzzedDataProvider &fuzzed_data) {
    mlir::Value val;
    bool is_float = false;
    if (fuzzed_data.ConsumeBool()) {  // return float tensor
      val = sampleFltTensorOp(fuzzed_data);
      is_float = true;
    } else {  // return int tensor
      val = sampleIntTensorOp(fuzzed_data);
      is_float = false;
    }
    return {val, is_float};
  }

  // Create a constant int op.
  mlir::Value createConstantIntOp(FuzzedDataProvider &fuzzed_data) {
    mlir::DenseIntElementsAttr constant_attr;
    switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 3)) {
      case 0:
        constant_attr = genDenseInt<8, int8_t>(fuzzed_data);
        break;
      case 1:
        constant_attr = genDenseInt<16, int16_t>(fuzzed_data);
        break;
      case 2:
        constant_attr = genDenseInt<32, int32_t>(fuzzed_data);
        break;
      case 3:
        constant_attr = genDenseInt<64, int64_t>(fuzzed_data);
        break;
      default:
        abort();
    }
    auto constant_int_Op =
        builder.create<mlir::spu::pphlo::ConstantOp>(loc, constant_attr);
    if (fuzzed_data.ConsumeBool()) {
      // Create Secret Type:
      // 1. Create a public type constant op;
      // 2. Utilize pphlo_convert op to convert public type into secret type
      auto mlir_ty =
          mlir::dyn_cast<mlir::RankedTensorType>(constant_int_Op.getType());
      mlir::spu::pphlo::TypeTools tool(&ctx);
      auto express_type = tool.getExpressedType(mlir_ty);
      mlir::Type ranked_tensor_type =
          tool.getType(express_type, mlir::spu::pphlo::Visibility::SECRET);
      addIntTensor(
          constant_int_Op);  // add the public type int tensor into the list
      return addIntTensor(builder.create<mlir::spu::pphlo::ConvertOp>(
          loc, ranked_tensor_type, constant_int_Op));
    } else {  // create Public Type
      return addIntTensor(constant_int_Op);
    }
  }

  // Create a constant int op with specific shape.
  mlir::Value createConstantIntOp(FuzzedDataProvider &fuzzed_data,
                                  const spu::Shape &shape) {
    mlir::DenseIntElementsAttr constant_attr;
    switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 3)) {
      case 0:
        constant_attr = genDenseInt<8, int8_t>(fuzzed_data, shape);
        break;
      case 1:
        constant_attr = genDenseInt<16, int16_t>(fuzzed_data, shape);
        break;
      case 2:
        constant_attr = genDenseInt<32, int32_t>(fuzzed_data, shape);
        break;
      case 3:
        constant_attr = genDenseInt<64, int64_t>(fuzzed_data, shape);
        break;
      default:
        abort();
    }
    auto constant_int_Op =
        builder.create<mlir::spu::pphlo::ConstantOp>(loc, constant_attr);
    if (fuzzed_data.ConsumeBool()) {  // create Secret Type
      auto mlir_ty =
          mlir::dyn_cast<mlir::RankedTensorType>(constant_int_Op.getType());
      mlir::spu::pphlo::TypeTools tool(&ctx);
      auto express_type = tool.getExpressedType(mlir_ty);
      mlir::Type ranked_tensor_type =
          tool.getType(express_type, mlir::spu::pphlo::Visibility::SECRET);
      addIntTensor(constant_int_Op);
      return addIntTensor(builder.create<mlir::spu::pphlo::ConvertOp>(
          loc, ranked_tensor_type, constant_int_Op));
    } else {  // create Public Type
      return addIntTensor(constant_int_Op);
    }
  }

  // Create a constant float op.
  mlir::Value createConstantFloatOp(FuzzedDataProvider &fuzzed_data) {
    mlir::DenseFPElementsAttr constant_attr;
    // SPU doesn't support 128bits float type.(2022-11-12)
    switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 1)) {
      case 0:
        constant_attr = genDenseFloat<32, float>(fuzzed_data);
        break;
      case 1:
        constant_attr = genDenseFloat<64, double>(fuzzed_data);
        break;
      default:
        abort();
    }
    auto constant_flt_op =
        builder.create<mlir::spu::pphlo::ConstantOp>(loc, constant_attr);
    if (fuzzed_data.ConsumeBool()) {  // create Secret Type
      auto mlir_ty =
          mlir::dyn_cast<mlir::RankedTensorType>(constant_flt_op.getType());
      mlir::spu::pphlo::TypeTools tool(&ctx);
      auto express_type = tool.getExpressedType(mlir_ty);
      auto ranked_tensor_type =
          tool.getType(express_type, mlir::spu::pphlo::Visibility::SECRET);
      addFltTensor(constant_flt_op);
      return addFltTensor(builder.create<mlir::spu::pphlo::ConvertOp>(
          loc, ranked_tensor_type, constant_flt_op));
    } else {  // create Public Type
      return addFltTensor(constant_flt_op);
    }
  }

  // Create a constant float/int op with specific shape.
  mlir::Value createConstantOp(FuzzedDataProvider &fuzzed_data,
                               const spu::Shape &shape) {
    if (fuzzed_data.ConsumeBool()) {  // return Float tensor op
      return createConstantFloatOp(fuzzed_data, shape);
    } else {  // return Int tensor op
      return createConstantIntOp(fuzzed_data, shape);
    }
  }

  // Create a constant float op with specific shape.
  mlir::Value createConstantFloatOp(FuzzedDataProvider &fuzzed_data,
                                    const spu::Shape &shape) {
    mlir::DenseFPElementsAttr constant_attr;
    // SPU doesn't support 128bits float type.(2022-11-12)
    switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(0, 1)) {
      case 0:
        constant_attr = genDenseFloat<32, float>(fuzzed_data, shape);
        break;
      case 1:
        constant_attr = genDenseFloat<64, double>(fuzzed_data, shape);
        break;
      default:
        abort();
    }
    auto constant_flt_op =
        builder.create<mlir::spu::pphlo::ConstantOp>(loc, constant_attr);
    if (fuzzed_data.ConsumeBool()) {  // create Secret Type
      auto mlir_ty =
          mlir::dyn_cast<mlir::RankedTensorType>(constant_flt_op.getType());
      mlir::spu::pphlo::TypeTools tool(&ctx);
      auto express_type = tool.getExpressedType(mlir_ty);
      auto ranked_tensor_type =
          tool.getType(express_type, mlir::spu::pphlo::Visibility::SECRET);
      addFltTensor(constant_flt_op);
      return addFltTensor(builder.create<mlir::spu::pphlo::ConvertOp>(
          loc, ranked_tensor_type, constant_flt_op));
    } else {  // create Public Type
      return addFltTensor(constant_flt_op);
    }
  }

  // Create UnaryElementwiseOp
  template <typename OpTy>
  void createUnaryElementwiseOp(FuzzedDataProvider &fuzzed_data) {
    if constexpr (isIntTensorOp<OpTy>()) {
      mlir::Value val = sampleIntTensorOp(fuzzed_data);  // get int tensor op.
      addIntTensor(builder.create<OpTy>(loc, val.getType(), val));
    } else if constexpr (isFloatTensorOp<OpTy>()) {  // get float tensor op.
      mlir::Value val = sampleFltTensorOp(fuzzed_data);
      addFltTensor(builder.create<OpTy>(loc, val.getType(), val));
    } else if constexpr (isAnyTensorOp<OpTy>()) {  // get random int or float
                                                   // tensor op.
      auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);
      if (is_float) {
        addFltTensor(builder.create<OpTy>(loc, val.getType(), val));
      } else {
        addIntTensor(builder.create<OpTy>(loc, val.getType(), val));
      }
    } else {
      printf(
          "invalid unary op, add this operation to isIntTensorOp, "
          "isFloatTensorOp or isAnyTensorOp\n");
      abort();
    }
  }

  // The shape of two operands of binary element wise op should be the same.
  template <typename OpTy>
  void createBinaryElementwiseOp(FuzzedDataProvider &fuzzed_data) {
    auto [lhs, lhs_is_float] = sampleAnyTensorOp(fuzzed_data);
    auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto lhs_shape = lhs_type.getShape();

    mlir::Value rhs;
    auto probability = fuzzed_data.ConsumeProbability<float>();
    // 75% probability creates new tensor op, 25% probability uses exist tensor
    // op.
    if (probability < 0.75) {
      rhs = createConstantOp(fuzzed_data, lhs_shape);
    } else {
      llvm::SmallVector<mlir::Value> rhs_list;
      for (auto i : int_tensor_values) {
        if (mlir::dyn_cast<mlir::RankedTensorType>(i.getType()).getShape() ==
            lhs_shape) {
          rhs_list.push_back(i);
        }
      }
      for (auto i : flt_tensor_values) {
        if (mlir::dyn_cast<mlir::RankedTensorType>(i.getType()).getShape() ==
            lhs_shape) {
          rhs_list.push_back(i);
        }
      }
      if (!rhs_list.empty()) {
        rhs = rhs_list[fuzzed_data.ConsumeIntegralInRange<uint16_t>(
            0, rhs_list.size() - 1)];
      } else {
        rhs = createConstantOp(fuzzed_data, lhs_shape);
      }
    }
    std::pair<mlir::Type, bool> element_type_res = genElementType(fuzzed_data);
    mlir::Type element_type = element_type_res.first;
    bool result_type_is_float = element_type_res.second;
    mlir::RankedTensorType ranked_tensor_type;
    mlir::spu::pphlo::TypeTools tools(&ctx);
    if (fuzzed_data.ConsumeBool()) {
      ranked_tensor_type = mlir::RankedTensorType::get(lhs_shape, element_type);
    } else {
      ranked_tensor_type = mlir::RankedTensorType::get(
          lhs_shape, mlir::spu::pphlo::SecretType::get(element_type));
    }
    if (result_type_is_float) {
      addFltTensor(builder.create<OpTy>(loc, ranked_tensor_type, lhs, rhs));
    } else {
      addIntTensor(builder.create<OpTy>(loc, ranked_tensor_type, lhs, rhs));
    }
  }

  template <typename OpTy>
  void createBinaryLogicalElementwiseOp(FuzzedDataProvider &fuzzed_data) {
    auto lhs = sampleIntTensorOp(fuzzed_data);
    auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto lhs_shape = lhs_type.getShape();

    mlir::Value rhs;
    auto probability = fuzzed_data.ConsumeProbability<float>();
    // 75% probability creates new tensor op, 25% probability uses exist tensor
    // op.
    if (probability < 0.75) {
      rhs = createConstantIntOp(fuzzed_data, lhs_shape);
    } else {
      llvm::SmallVector<mlir::Value> rhs_list;
      for (auto i : int_tensor_values) {
        if (mlir::dyn_cast<mlir::RankedTensorType>(i.getType()).getShape() ==
            lhs_shape) {
          rhs_list.push_back(i);
        }
      }
      if (!rhs_list.empty()) {
        rhs = rhs_list[fuzzed_data.ConsumeIntegralInRange<uint16_t>(
            0, rhs_list.size() - 1)];
      } else {
        rhs = createConstantIntOp(fuzzed_data, lhs_shape);
      }
    }
    mlir::Type element_type = genIntElementType(fuzzed_data);
    mlir::RankedTensorType ranked_tensor_type;
    if (fuzzed_data.ConsumeBool()) {
      ranked_tensor_type = mlir::RankedTensorType::get(lhs_shape, element_type);
    } else {
      ranked_tensor_type = mlir::RankedTensorType::get(
          lhs_shape, mlir::spu::pphlo::SecretType::get(element_type));
    }
    addIntTensor(builder.create<OpTy>(loc, ranked_tensor_type, lhs, rhs));
  }

  template <typename OpTy>
  void createComparisonOp(FuzzedDataProvider &fuzzed_data) {
    auto [lhs, lhs_is_float] = sampleAnyTensorOp(fuzzed_data);
    auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto lhs_shape = lhs_type.getShape();

    mlir::Value rhs;
    auto probability = fuzzed_data.ConsumeProbability<float>();
    // 75% probability creates new tensor op, 25% probability uses exist tensor
    // op.
    if (probability < 0.75) {
      rhs = createConstantOp(fuzzed_data, lhs_shape);
    } else {
      llvm::SmallVector<mlir::Value> rhs_list;
      for (auto i : int_tensor_values) {
        if (mlir::dyn_cast<mlir::RankedTensorType>(i.getType()).getShape() ==
            lhs_shape) {
          rhs_list.push_back(i);
        }
      }
      for (auto i : flt_tensor_values) {
        if (mlir::dyn_cast<mlir::RankedTensorType>(i.getType()).getShape() ==
            lhs_shape) {
          rhs_list.push_back(i);
        }
      }
      if (!rhs_list.empty()) {
        rhs = rhs_list[fuzzed_data.ConsumeIntegralInRange<uint16_t>(
            0, rhs_list.size() - 1)];
      } else {
        rhs = createConstantOp(fuzzed_data, lhs_shape);
      }
    }
    mlir::Type element_type = genIntElementType(fuzzed_data);
    mlir::RankedTensorType ranked_tensor_type;
    if (fuzzed_data.ConsumeBool()) {
      ranked_tensor_type = mlir::RankedTensorType::get(lhs_shape, element_type);
    } else {
      ranked_tensor_type = mlir::RankedTensorType::get(
          lhs_shape, mlir::spu::pphlo::SecretType::get(element_type));
    }

    addIntTensor(builder.create<OpTy>(loc, ranked_tensor_type, lhs, rhs));
  }

  void createBroadCastOp(
      FuzzedDataProvider &fuzzed_data) {  // TODO: Haven't finish this op.
    auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);

    // expand dimension shape
    // TODO: better strategy
    std::vector<int64_t> int_repr;
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    for (auto dim_elems : ty.getShape()) {
      int_repr.push_back(dim_elems +
                         (fuzzed_data.ConsumeIntegral<int64_t>() % 10));
    }
    auto broadcast_dim =
        mlir::DenseI64ArrayAttr::get(builder.getContext(), int_repr);

    using OpTy = mlir::spu::pphlo::BroadcastOp;
    if (is_float) {
      addFltTensor(
          builder.create<OpTy>(loc, val.getType(), val, broadcast_dim));
    } else {  // NOLINT(llvm-else-after-return)
      addIntTensor(
          builder.create<OpTy>(loc, val.getType(), val, broadcast_dim));
    }
  }

  void createTransposeOp(FuzzedDataProvider &fuzzed_data) {
    auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);

    std::vector<int64_t> int_repr;
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    for (auto dim_elems : ty.getShape()) {
      (void)dim_elems;
      int_repr.push_back(fuzzed_data.ConsumeIntegral<int64_t>());
    }
    auto broadcast_dim =
        mlir::DenseI64ArrayAttr::get(builder.getContext(), int_repr);

    using OpTy = mlir::spu::pphlo::TransposeOp;
    if (is_float) {
      addFltTensor(
          builder.create<OpTy>(loc, val.getType(), val, broadcast_dim));
    } else {
      addIntTensor(
          builder.create<OpTy>(loc, val.getType(), val, broadcast_dim));
    }
  }

  void createReverseOp(FuzzedDataProvider &fuzzed_data) {
    auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);

    std::vector<int64_t> int_repr;
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    for (auto dim_elems : ty.getShape()) {
      (void)dim_elems;
      int_repr.push_back(fuzzed_data.ConsumeIntegral<int64_t>());
    }
    auto broadcast_dim =
        mlir::DenseI64ArrayAttr::get(builder.getContext(), int_repr);

    using OpTy = mlir::spu::pphlo::ReverseOp;
    if (is_float) {
      addFltTensor(
          builder.create<OpTy>(loc, val.getType(), val, broadcast_dim));
    } else {  // NOLINT(llvm-else-after-return)
      addIntTensor(
          builder.create<OpTy>(loc, val.getType(), val, broadcast_dim));
    }
  }

  void createSliceOp(FuzzedDataProvider &fuzzed_data) {
    // TODO: Haven't successfully construct this op.
    auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);

    // expand dimension shape
    // TODO: better strategy
    std::vector<int64_t> int_repr;
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    for (auto dim_elems : ty.getShape()) {
      (void)dim_elems;
      int_repr.push_back(fuzzed_data.ConsumeIntegral<int64_t>());
    }
    auto broadcast_dim =
        mlir::DenseI64ArrayAttr::get(builder.getContext(), int_repr);

    using OpTy = mlir::spu::pphlo::SliceOp;
    if (is_float) {
      addFltTensor(builder.create<OpTy>(loc, val.getType(), val, broadcast_dim,
                                        broadcast_dim, broadcast_dim));
    } else {  // NOLINT(llvm-else-after-return)
      addIntTensor(builder.create<OpTy>(loc, val.getType(), val, broadcast_dim,
                                        broadcast_dim, broadcast_dim));
    }
  }

  void createConcatenateOp(FuzzedDataProvider &fuzzed_data) {
    // TODO: concat different tensors, current implementation only concatenate
    // the same tensor
    uint64_t mx_dim = 0;
    mlir::Value val;
    bool is_float;
    do {
      auto [ret_val, ret_is_float] = sampleAnyTensorOp(fuzzed_data);
      mx_dim = mlir::dyn_cast<mlir::RankedTensorType>(ret_val.getType())
                   .getShape()
                   .size();
      val = ret_val;
      is_float = ret_is_float;
    } while (mx_dim == 0);

    std::vector<mlir::Value> concat_tensors;
    auto ext_sz = fuzzed_data.ConsumeIntegralInRange<uint8_t>(1, 50);
    for (int i = 0; i < ext_sz; i++) {
      concat_tensors.push_back(val);
    }

    // select one of concat dimension shape
    int64_t concat_dim_val =
        fuzzed_data.ConsumeIntegralInRange<uint16_t>(0, mx_dim - 1);
    mlir::IntegerAttr concat_dim = builder.getI64IntegerAttr(concat_dim_val);

    auto old_ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    auto elem_ty = old_ty.getElementType();
    auto old_shape = old_ty.getShape();
    int64_t new_shape_val = old_shape[concat_dim_val] * ext_sz;
    std::vector<int64_t> new_shape(old_shape.begin(), old_shape.end());
    new_shape[concat_dim_val] = new_shape_val;

    mlir::RankedTensorType rankedTensorType =
        mlir::RankedTensorType::get(new_shape, elem_ty);
    using OpTy = mlir::spu::pphlo::ConcatenateOp;
    if (is_float) {
      addFltTensor(builder.create<OpTy>(loc, rankedTensorType, concat_tensors,
                                        concat_dim));
    } else {  // NOLINT(llvm-else-after-return)
      addIntTensor(builder.create<OpTy>(loc, rankedTensorType, concat_tensors,
                                        concat_dim));
    }
  }

  mlir::spu::pphlo::SelectOp createSelectOp(FuzzedDataProvider &fuzzed_data) {
    using OpTy = mlir::spu::pphlo::SelectOp;

    auto cond_tensor = sampleIntTensorOp(fuzzed_data);
    auto [true_tensor, is_float] = sampleAnyTensorOp(fuzzed_data);
    if (is_float) {
      auto false_tensor = sampleFltTensorOp(fuzzed_data);
      return addFltTensor(builder.create<OpTy>(
          loc, true_tensor.getType(), cond_tensor, true_tensor, false_tensor));
    } else {
      auto false_tensor = sampleIntTensorOp(fuzzed_data);
      return addIntTensor(builder.create<OpTy>(
          loc, true_tensor.getType(), cond_tensor, true_tensor, false_tensor));
    }
  }

  void createIotaOp(FuzzedDataProvider &fuzzed_data) {
    using OpTy = mlir::spu::pphlo::IotaOp;

    auto iota_dimension =
        builder.getI64IntegerAttr(fuzzed_data.ConsumeIntegral<int64_t>());
    std::pair<mlir::RankedTensorType, bool> result =
        genShapedPPHloTensorType(fuzzed_data);
    if (result.second) {
      addFltTensor(builder.create<OpTy>(loc, result.first, iota_dimension));
    } else {
      addIntTensor(builder.create<OpTy>(loc, result.first, iota_dimension));
    }
  }

  void createEpsilonOp(FuzzedDataProvider &fuzzed_data) {
    using OpTy = mlir::spu::pphlo::EpsilonOp;

    mlir::Type element_type = genFPElementType(fuzzed_data);
    auto shape = genShape(fuzzed_data);
    mlir::RankedTensorType ranked_tensor_type;
    if (fuzzed_data.ConsumeBool()) {
      ranked_tensor_type = mlir::RankedTensorType::get(shape, element_type);
    } else {
      ranked_tensor_type = mlir::RankedTensorType::get(
          shape, mlir::spu::pphlo::SecretType::get(element_type));
    }

    addFltTensor(builder.create<OpTy>(loc, ranked_tensor_type));
  }

  void createConvertOp(FuzzedDataProvider &fuzzed_data) {
    using OpTy = mlir::spu::pphlo::ConvertOp;

    auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);
    auto shape =
        mlir::dyn_cast<mlir::RankedTensorType>(val.getType()).getShape();
    std::pair<mlir::Type, bool> element_type_res = genElementType(fuzzed_data);
    mlir::Type element_type = element_type_res.first;
    bool result_type_is_float = element_type_res.second;
    mlir::RankedTensorType ranked_tensor_type;
    if (fuzzed_data.ConsumeBool()) {
      ranked_tensor_type = mlir::RankedTensorType::get(shape, element_type);
    } else {
      ranked_tensor_type = mlir::RankedTensorType::get(
          shape, mlir::spu::pphlo::SecretType::get(element_type));
    }
    if (result_type_is_float) {
      addFltTensor(builder.create<OpTy>(loc, ranked_tensor_type, val));
    } else {
      addIntTensor(builder.create<OpTy>(loc, ranked_tensor_type, val));
    }
  }

  // Function to find the factors of a number
  std::vector<int64_t> findFactors(int n) {
    std::vector<int64_t> factors;

    // Loop through all numbers from 2 to sqrt(n)
    for (int i = 2; i <= sqrt(n); i++) {
      // If i is a factor of n, add it to the list of factors
      if (n % i == 0) {
        factors.push_back(i);
        // If i is not the square root of n, add the corresponding factor n/i
        if (i != n / i) {
          factors.push_back(n / i);
        }
      }
    }

    // Add 1 and n to the list of factors
    factors.push_back(1);
    factors.push_back(n);

    return factors;
  }

  void findCombinations(std::vector<std::vector<int64_t>> &combinations,
                        std::vector<int64_t> &currentCombination,
                        const std::vector<int64_t> &factors, int n, int k,
                        int startIndex) {
    if (k == 0) {
      int64_t product = 1;
      for (size_t i = 0; i < currentCombination.size(); i++) {
        product *= currentCombination[i];
      }
      if (product == n) {
        combinations.push_back(currentCombination);
      }
      return;
    }

    for (size_t i = startIndex; i <= factors.size() - k; i++) {
      currentCombination[currentCombination.size() - k] = factors[i];
      findCombinations(combinations, currentCombination, factors, n, k - 1,
                       i + 1);
    }
  }

  std::vector<std::vector<int64_t>> findNumbers(int n) {
    std::vector<int64_t> factors = findFactors(n);
    std::vector<std::vector<int64_t>> combinations;

    for (size_t k = 1; k <= factors.size(); k++) {
      std::vector<int64_t> currentCombination(k);
      findCombinations(combinations, currentCombination, factors, n, k, 0);
    }

    return combinations;
  }

  void createReshapeOp(FuzzedDataProvider &fuzzed_data) {
    using OpTy = mlir::spu::pphlo::ReshapeOp;

    auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);
    auto mlir_ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    auto element_type = mlir_ty.getElementType();
    spu::Shape shape = mlir_ty.getShape();
    auto new_shape_combinations = findNumbers(shape.numel());
    auto new_shape =
        new_shape_combinations[rand() % new_shape_combinations.size()];
    mlir::RankedTensorType ranked_tensor_type =
        mlir::RankedTensorType::get(new_shape, element_type);
    if (is_float) {
      addFltTensor(builder.create<OpTy>(loc, ranked_tensor_type, val));
    } else {
      addIntTensor(builder.create<OpTy>(loc, ranked_tensor_type, val));
    }
  }

  mlir::Value createConstantOpWithType(FuzzedDataProvider &fuzzed_data,
                                       mlir::RankedTensorType mlir_ty) {
    using OpTy = mlir::spu::pphlo::ConstantOp;
    mlir::spu::pphlo::TypeTools tool(mlir_ty.getContext());
    auto expressed_type = tool.getExpressedType(mlir_ty);
    bool is_float = mlir::isa<mlir::FloatType>(expressed_type);
    unsigned int bit_width;
    if (is_float) {
      bit_width = mlir::dyn_cast<mlir::FloatType>(expressed_type)
                      .getIntOrFloatBitWidth();
    } else {
      bit_width = mlir::dyn_cast<mlir::IntegerType>(expressed_type)
                      .getIntOrFloatBitWidth();
    }
    auto visibility = tool.getTypeVisibility(mlir_ty);

    auto new_shape = genShape(fuzzed_data);
    mlir::RankedTensorType ranked_tensor_type_wo_visibility =
        mlir::RankedTensorType::get(new_shape, expressed_type);
    int64_t elements_num = new_shape.numel();

    mlir::DenseElementsAttr constant_attr;
    if (is_float) {
      if (bit_width == 32) {
        std::vector<float> values(elements_num);
        for (int i = 0; i < elements_num; i++) {
          values[i] = fuzzed_data.ConsumeFloatingPoint<float>();
        }
        constant_attr = mlir::DenseFPElementsAttr::get(
            ranked_tensor_type_wo_visibility, values);
      } else if (bit_width == 64) {
        std::vector<double> values(elements_num);
        for (int i = 0; i < elements_num; i++) {
          values[i] = fuzzed_data.ConsumeFloatingPoint<double>();
        }
        constant_attr = mlir::DenseFPElementsAttr::get(
            ranked_tensor_type_wo_visibility, values);
      } else {
        abort();
      }
    } else {
      if (bit_width == 8) {
        std::vector<int8_t> values(elements_num);
        for (int i = 0; i < elements_num; i++) {
          values[i] = fuzzed_data.ConsumeIntegral<int8_t>();
        }
        constant_attr = mlir::DenseIntElementsAttr::get(
            ranked_tensor_type_wo_visibility, values);
      } else if (bit_width == 16) {
        std::vector<int16_t> values(elements_num);
        for (int i = 0; i < elements_num; i++) {
          values[i] = fuzzed_data.ConsumeIntegral<int16_t>();
        }
        constant_attr = mlir::DenseIntElementsAttr::get(
            ranked_tensor_type_wo_visibility, values);
      } else if (bit_width == 32) {
        std::vector<int32_t> values(elements_num);
        for (int i = 0; i < elements_num; i++) {
          values[i] = fuzzed_data.ConsumeIntegral<int32_t>();
        }
        constant_attr = mlir::DenseIntElementsAttr::get(
            ranked_tensor_type_wo_visibility, values);
      } else if (bit_width == 64) {
        std::vector<int64_t> values(elements_num);
        for (int i = 0; i < elements_num; i++) {
          values[i] = fuzzed_data.ConsumeIntegral<int64_t>();
        }
        constant_attr = mlir::DenseIntElementsAttr::get(
            ranked_tensor_type_wo_visibility, values);
      } else {
        abort();
      }
    }

    auto constant_op = builder.create<OpTy>(loc, constant_attr);
    if (visibility == mlir::spu::pphlo::Visibility::SECRET) {
      mlir::RankedTensorType sec_ranked_tensor_type =
          mlir::RankedTensorType::get(
              new_shape, mlir::spu::pphlo::SecretType::get(expressed_type));
      if (is_float) {
        addFltTensor(constant_op);
        return addFltTensor(builder.create<mlir::spu::pphlo::ConvertOp>(
            loc, sec_ranked_tensor_type, constant_op));
      } else {
        addIntTensor(constant_op);
        return addIntTensor(builder.create<mlir::spu::pphlo::ConvertOp>(
            loc, sec_ranked_tensor_type, constant_op));
      }
    } else if (visibility == mlir::spu::pphlo::Visibility::PUBLIC) {
      if (is_float) {
        return addFltTensor(constant_op);
      } else {
        return addIntTensor(constant_op);
      }
    } else {
      abort();
    }
  }

  void createRngOp(FuzzedDataProvider &fuzzed_data) {
    // TODO: RNG op doesn't support secret type.
    using OpTy = mlir::spu::pphlo::RngOp;
    auto [lhs, lhs_is_float] = sampleAnyTensorOp(fuzzed_data);
    auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    if (lhs_type == nullptr) {
      std::cout << "LHS Type dyn_cast failed." << std::endl;
    }
    mlir::spu::pphlo::TypeTools tool(lhs.getContext());
    auto visibility = tool.getTypeVisibility(lhs_type);
    if (visibility == mlir::spu::pphlo::Visibility::PUBLIC) {
      std::cout << "Construct RNG Op" << std::endl;
      mlir::Value rhs;
      auto probability = fuzzed_data.ConsumeProbability<float>();
      // 75% probability creates new tensor op, 25% probability uses exist
      // tensor op.
      if (probability < 0.75) {
        rhs = createConstantOpWithType(fuzzed_data, lhs_type);
      } else {
        llvm::SmallVector<mlir::Value> rhs_list;
        if (lhs_is_float) {
          for (auto i : flt_tensor_values) {
            if (mlir::dyn_cast<mlir::RankedTensorType>(i.getType()) ==
                lhs_type) {
              rhs_list.push_back(i);
            }
          }
        } else {
          for (auto i : int_tensor_values) {
            if (mlir::dyn_cast<mlir::RankedTensorType>(i.getType()) ==
                lhs_type) {
              rhs_list.push_back(i);
            }
          }
        }

        if (!rhs_list.empty()) {
          rhs = rhs_list[fuzzed_data.ConsumeIntegralInRange<uint16_t>(
              0, rhs_list.size() - 1)];
        } else {
          rhs = createConstantOpWithType(fuzzed_data, lhs_type);
        }
      }

      auto res_shape = genShape(fuzzed_data);
      auto expressed_type = tool.getExpressedType(lhs_type);
      mlir::RankedTensorType res_type =
          mlir::RankedTensorType::get(res_shape, expressed_type);

      if (lhs_is_float) {
        addFltTensor(builder.create<OpTy>(loc, res_type, lhs, rhs));
      } else {
        addIntTensor(builder.create<OpTy>(loc, res_type, lhs, rhs));
      }
    }
  }

  void createDotOp(FuzzedDataProvider &fuzzed_data) {
    using OpTy = mlir::spu::pphlo::DotOp;

    auto [val, is_float] = sampleAnyTensorOp(fuzzed_data);
    auto [val_new, is_float_new] = sampleAnyTensorOp(fuzzed_data);
    if (is_float) {
      addFltTensor(builder.create<OpTy>(loc, val.getType(), val, val_new));
    } else {
      addIntTensor(builder.create<OpTy>(loc, val.getType(), val, val_new));
    }
  }

  enum {
#define mDefOpGen(Op) GenInd##Op,

#include "GenOps.h"

    GenIndBroadcastOp,
    GenIndConcatenateOp,
    GenIndSelectOp,
    GenIndIotaOp,
    GenIndEpsilonOp,
    GenIndConvertOp,
    GenIndReshapeOp,
    GenIndRngOp,
    GenIndDotOp,
    GenIndTransposeOp,
    GenIndReverseOp,
    GenIndSliceOp,
    GenIndSz,
  };

  void createRandomOp(FuzzedDataProvider &fuzzed_data) {
    auto rnd_idx = fuzzed_data.ConsumeIntegralInRange<uint8_t>(
        0, GenIndSz - 1);  // The max depends on the last element of the enum.
    switch (rnd_idx) {
#define mDefOpGen(Op)        \
  case GenInd##Op:           \
    create##Op(fuzzed_data); \
    break;
#define mDefOpGenUnary(Op)                                       \
  case GenInd##Op:                                               \
    createUnaryElementwiseOp<mlir::spu::pphlo::Op>(fuzzed_data); \
    break;
#define mDefOpGenBinary(Op)                                       \
  case GenInd##Op:                                                \
    createBinaryElementwiseOp<mlir::spu::pphlo::Op>(fuzzed_data); \
    break;
#define mDefOpGenBinaryLogical(Op)                                       \
  case GenInd##Op:                                                       \
    createBinaryLogicalElementwiseOp<mlir::spu::pphlo::Op>(fuzzed_data); \
    break;
#define mDefOpGenComparison(Op)                            \
  case GenInd##Op:                                         \
    createComparisonOp<mlir::spu::pphlo::Op>(fuzzed_data); \
    break;

#include "GenOps.h"

      case GenIndBroadcastOp:
        createBroadCastOp(fuzzed_data);
        break;
      case GenIndConcatenateOp:
        createConcatenateOp(fuzzed_data);
        break;
      case GenIndSelectOp:
        createSelectOp(fuzzed_data);
        break;
      case GenIndIotaOp:
        createIotaOp(fuzzed_data);
        break;
      case GenIndEpsilonOp:
        createEpsilonOp(fuzzed_data);
        break;
      case GenIndConvertOp:
        createConvertOp(fuzzed_data);
        break;
      case GenIndReshapeOp:
        createReshapeOp(fuzzed_data);
        break;
      case GenIndRngOp:
        createRngOp(fuzzed_data);
        break;
      case GenIndDotOp:
        createDotOp(fuzzed_data);
        break;
      case GenIndTransposeOp:
        createTransposeOp(fuzzed_data);
        break;
      case GenIndReverseOp:
        createReverseOp(fuzzed_data);
        break;
      case GenIndSliceOp:
        createSliceOp(fuzzed_data);
        break;
      default:
        //                module_op.dump();
        return;
    }
  }

  std::string mlirGeneration(FuzzedDataProvider &fuzzed_data) {
    loc = mlir::UnknownLoc::get(&ctx);
    module_op = builder.create<mlir::ModuleOp>(loc);
    builder.setInsertionPointToEnd(module_op.getBody());

    // construct fixed return/result type
    // TODO: construct random return/result type
    llvm::SmallVector<mlir::Type> ret_types;
    ret_types.push_back(mlir::RankedTensorType::get({}, builder.getI32Type()));
    auto func_ty = builder.getFunctionType(std::nullopt, ret_types);
    auto func_op =
        builder.create<mlir::func::FuncOp>(loc, llvm::StringRef("main"),
                                           func_ty);  // construct main function

    // add entry block and insert the following operations into the start of the
    // block
    auto *entry_block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    while (fuzzed_data.remaining_bytes() > 0) {
      createRandomOp(fuzzed_data);
    }

    // add ret constant op
    auto ret_const_op = builder.create<mlir::spu::pphlo::ConstantOp>(
        mlir::UnknownLoc::get(&ctx),
        mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get({}, builder.getI32Type()),
            {llvm::APInt(32, 5, true)}));
    builder.create<mlir::spu::pphlo::ReturnOp>(
        mlir::UnknownLoc::get(&ctx),
        mlir::ValueRange(ret_const_op.getResult()));

    // DEBUG usage
    std::string mlir_str;
    llvm::raw_string_ostream output(mlir_str);
    module_op.print(output);
    std::cout << mlir_str << std::endl;

    return mlir_str;
  }

 protected:
  mlir::MLIRContext &ctx;
  mlir::OpBuilder builder;
  mlir::UnknownLoc loc;
  mlir::ModuleOp module_op;

  llvm::SmallVector<mlir::Value> int_tensor_values;
  llvm::SmallVector<mlir::Value> flt_tensor_values;
  llvm::SmallVector<mlir::Value> dim_tensor_values[10];
};

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " [input_file]" << std::endl;
    return 0;
  }

  // read the whole file into the memory
  std::ifstream input_file(
      std::string(argv[1]),
      std::ios::in |
          std::ios::
              binary);  // TODO: fix the argv[3] error,
                        // https://github.com/bazelbuild/intellij/issues/2958
  input_file.seekg(0, std::ios::end);
  size_t input_file_len = input_file.tellg();
  input_file.seekg(0, std::ios::beg);
  //    std::cout << "file size: " << input_file_len << std::endl;
  char *bin_buf = new char[input_file_len];
  input_file.read(bin_buf, input_file_len);
  input_file.close();

  FuzzedDataProvider fuzzed_data(
      reinterpret_cast<uint8_t *>(bin_buf),
      input_file_len);  // Convert the input binary file to FuzzedDataProvider
                        // Structure

  // Load the Builtin and PPHLO Dialect
  auto ctx = createContext();
  Mutator r(*ctx);

  // Construct PPHLO Virtual Machine Execution Environment
  size_t world_size = 2;
  spu::FieldType field;
  spu::ProtocolKind protocol;
  switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(
      1, 4)) {  // Generating different protocols
    case 1:
      protocol = spu::REF2K;
      break;
    case 2:
      protocol = spu::SEMI2K;
      break;
    case 3:
      protocol = spu::ABY3;  // ABY3 protocol needs 3 worlds
      world_size = 3;
      break;
    case 4:
      protocol = spu::CHEETAH;
      break;
    default:
      abort();
  }
  switch (fuzzed_data.ConsumeIntegralInRange<uint8_t>(
      1, 3)) {  // Generating different fields
    case 1:
      field = spu::FM32;
      break;
    case 2:
      field = spu::FM64;
      break;
    case 3:
      field = spu::FM128;
      break;
    default:
      abort();
  }

  std::cout << "world_size: " << world_size << ", field: " << field
            << ", protocol: " << protocol << std::endl;  // DEBUG usage
  spu::device::pphlo::test::Runner fuzz_runner(world_size, field, protocol);
  try {
    std::string mlir_code =
        r.mlirGeneration(fuzzed_data);  // Generating the pphlo opcode mlir
    fuzz_runner.run(mlir_code);
    // TODO: add verify feature
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    delete[] bin_buf;
    return 1;
  }

  delete[] bin_buf;
  std::cout << "Successfully Executing" << std::endl;
  return 0;
}
