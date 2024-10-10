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

#include "libspu/kernel/hlo/builder.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/core/core.h"
#include "libspu/device/api.h"
#include "libspu/device/executor.h"
#include "libspu/dialect/pphlo/IR/base_enums.h"
#include "libspu/dialect/pphlo/IR/dialect.h"
#include "libspu/dialect/ring/IR/dialect.h"
#include "libspu/dialect/utils/utils.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/factory.h"

namespace spu::kernel::hlo {

namespace {

inline mlir::spu::pphlo::Visibility GetVisibilityType(spu::Visibility vtype) {
  if (vtype == VIS_PUBLIC) {
    return mlir::spu::pphlo::Visibility::PUBLIC;
  } else if (vtype == VIS_SECRET || vtype == VIS_PRIVATE) {
    return mlir::spu::pphlo::Visibility::SECRET;
  } else {
    SPU_ENFORCE(false, "should not be here");
    return {};
  }
}

inline mlir::Type GetElementType(mlir::MLIRContext *ctx, spu::PtType pt_type) {
  switch (pt_type) {
#define SWITCH_CASE_INT(_PtT_, _Bits_, _Sign_)                        \
  case _PtT_: {                                                       \
    return mlir::IntegerType::get(                                    \
        ctx, _Bits_, mlir::IntegerType::SignednessSemantics::_Sign_); \
  }

    SWITCH_CASE_INT(PT_I1, 1, Signless)
    SWITCH_CASE_INT(PT_U8, 8, Unsigned)
    SWITCH_CASE_INT(PT_I8, 8, Signless)
    SWITCH_CASE_INT(PT_U16, 16, Unsigned)
    SWITCH_CASE_INT(PT_I16, 16, Signless)
    SWITCH_CASE_INT(PT_U32, 32, Unsigned)
    SWITCH_CASE_INT(PT_I32, 32, Signless)
    SWITCH_CASE_INT(PT_U64, 64, Unsigned)
    SWITCH_CASE_INT(PT_I64, 64, Signless)
    SWITCH_CASE_INT(PT_I128, 128, Signless)

#undef SWITCH_CASE_INT

#define SWITCH_CASE_FLOAT(_PtT_, _MlirT_)      \
  case _PtT_: {                                \
    return mlir::FloatType::get##_MlirT_(ctx); \
  }

    SWITCH_CASE_FLOAT(PT_F16, F16)
    SWITCH_CASE_FLOAT(PT_F32, F32)
    SWITCH_CASE_FLOAT(PT_F64, F64)

#undef SWITCH_CASE_FLOAT

    default: {
      SPU_ENFORCE(false, "should not be here");
      return {};
    }
  }
}
}  // namespace

using namespace mlir::detail;
using namespace mlir::spu::pphlo;
using namespace spu::compiler;
using namespace spu::device;

HloBuilder::HloBuilder()
    : mlir_ctx_(),
      module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlir_ctx_))),
      builder_(mlir::OpBuilder(module_.getBodyRegion())),
      loc_(mlir::UnknownLoc::get(&mlir_ctx_)),
      type_tools_(&mlir_ctx_) {
  mlir_ctx_.loadDialect<PPHloDialect, mlir::func::FuncDialect,
                        mlir::arith::ArithDialect, mlir::spu::ring::RingDialect,
                        mlir::tensor::TensorDialect>();

  mlir::DialectRegistry registry;
  mlir_ctx_.appendDialectRegistry(registry);

  main_fun_op_ = mlir::func::FuncOp::create(loc_, "main",
                                            builder_.getFunctionType({}, {}));
  builder_.setInsertionPointToStart(main_fun_op_.addEntryBlock());

  module_.push_back(main_fun_op_);
}

void HloBuilder::compile(const std::vector<mlir::Value> &outputs) {
  Return(outputs);

#ifdef DEBUG
  printf("%s\n", mlir::spu::mlirObjectToString(module_).c_str());
#endif  // DEBUG

  spu::CompilerOptions opt;
  CompilationContext ctx(opt, &mlir_ctx_);

  Core(&ctx).doit(module_);

#ifdef DEBUG
  printf("%s\n", mlir::spu::mlirObjectToString(module_).c_str());
#endif  // DEBUG
}

std::vector<spu::MemRef> HloBuilder::execute(spu::SPUContext *spu_ctx,
                                             std::vector<spu::MemRef> params) {
  ConcreteExecutor executor;
  ExecutionOptions opts;

  return runRegion(&executor, spu_ctx, nullptr, main_fun_op_.getBody(),
                   {params.data(), params.size()}, opts);
}

std::string HloBuilder::EmitCodes() const {
  return mlir::spu::mlirObjectToString(module_);
}

mlir::Value HloBuilder::Argument(spu::PtType pt_type,
                                 spu::Visibility visibility,
                                 const Shape &shape) {
  auto ranked_type =
      mlir::RankedTensorType::get(shape, GetElementType(&mlir_ctx_, pt_type));
  auto arg_type = type_tools_.getType(
      ranked_type, visibility == spu::Visibility::VIS_PUBLIC
                       ? mlir::spu::pphlo::Visibility::PUBLIC
                       : mlir::spu::pphlo::Visibility::SECRET);

  auto arg_type_list = llvm::to_vector(main_fun_op_.getArgumentTypes());
  arg_type_list.emplace_back(arg_type);

  main_fun_op_.setFunctionType(builder_.getFunctionType(arg_type_list, {}));

  return main_fun_op_.getBody().addArgument(arg_type, loc_);
}

mlir::Value HloBuilder::SplatConstant(const PtBufferView &view,
                                      const mlir::Value &as_shape) {
  SPU_ENFORCE(view.shape.numel() == 1, "Only scalar is allowed");
  mlir::TypedAttr const_attr;
  auto pt_type = view.pt_type;
  mlir::Type element_type;

  switch (pt_type) {
#define SWITCH_CASE_INT_AUX(_PtT_, _BuiltinT_, _Bits_)                  \
  case _PtT_: {                                                         \
    element_type = mlir::IntegerType::get(&mlir_ctx_, _Bits_);          \
    const_attr =                                                        \
        builder_.getIntegerAttr(element_type, view.get<_BuiltinT_>(0)); \
    break;                                                              \
  }

#define SWITCH_CASE_INT(_PtT_, _BuiltinT_) \
  SWITCH_CASE_INT_AUX(_PtT_, _BuiltinT_, sizeof(_BuiltinT_) * 8)

    SWITCH_CASE_INT_AUX(PT_I1, bool, 1)
    SWITCH_CASE_INT(PT_I8, int8_t)
    SWITCH_CASE_INT(PT_U8, uint8_t)
    SWITCH_CASE_INT(PT_I16, int16_t)
    SWITCH_CASE_INT(PT_U16, uint16_t)
    SWITCH_CASE_INT(PT_I32, int32_t)
    SWITCH_CASE_INT(PT_U32, uint32_t)
    SWITCH_CASE_INT(PT_I64, int64_t)
    SWITCH_CASE_INT(PT_U64, uint64_t)
    SWITCH_CASE_INT(PT_I128, int128_t)

#undef SWITCH_CASE_INT
#undef SWITCH_CASE_INT_AUX

#define SWITCH_CASE_FLOAT(_PtT_, _BuiltinT_, _MlirT_)                   \
  case _PtT_: {                                                         \
    auto element_type = mlir::FloatType::get##_MlirT_(&mlir_ctx_);      \
    const_attr =                                                        \
        builder_.getIntegerAttr(element_type, view.get<_BuiltinT_>(0)); \
    break;                                                              \
  }

    SWITCH_CASE_FLOAT(PT_F16, half_float::half, F16)
    SWITCH_CASE_FLOAT(PT_F32, float, F32)
    SWITCH_CASE_FLOAT(PT_F64, double, F64)

#undef SWITCH_CASE_FLOAT

    default: {
      SPU_ENFORCE(false, "should not be here");
      return {};
    }
  }

  auto const_value =
      mlir::spu::splatifyConstant(builder_, const_attr, as_shape);

  switch (pt_type) {
    case PT_U8:
      [[fallthrough]];
    case PT_U16:
      [[fallthrough]];
    case PT_U32:
      [[fallthrough]];
    case PT_U64: {
      auto unsigned_type =
          builder_.getIntegerType(element_type.getIntOrFloatBitWidth(), false);
      auto current_type = mlir::cast<mlir::ShapedType>(const_value.getType());
      const_value = builder_.create<mlir::spu::pphlo::BitcastConvertOp>(
          loc_, current_type.clone(unsigned_type), const_value);
    }
    default:
      // noop
      break;
  }

  return const_value;
}

mlir::Value HloBuilder::Splat(const mlir::Value &in,
                              const mlir::Value &as_shape) {
  mlir::Value element = in;
  if (mlir::dyn_cast<mlir::ShapedType>(in.getType())) {
    element =
        builder_.create<mlir::tensor::ExtractOp>(loc_, in, mlir::ValueRange());
  }

  llvm::SmallVector<mlir::Value> dynamic_dim;
  auto base_shape = mlir::cast<mlir::ShapedType>(as_shape.getType()).getShape();

  if (mlir::ShapedType::isDynamicShape(base_shape)) {
    for (size_t rank = 0; rank < base_shape.size(); ++rank) {
      if (mlir::ShapedType::isDynamic(base_shape[rank])) {
        dynamic_dim.emplace_back(
            builder_.create<mlir::tensor::DimOp>(loc_, as_shape, rank));
      }
    }
  }

  return builder_.create<mlir::tensor::SplatOp>(loc_, element, base_shape,
                                                dynamic_dim);
}

mlir::Value HloBuilder::Constant(const PtBufferView &view,
                                 const Shape &out_shape) {
  auto data = view.ptr;
  auto pt_type = view.pt_type;

  auto shape = view.shape;
  size_t numel = shape.numel();

  if (shape.size() == 0UL) {
    shape = out_shape;
  }

  mlir::Value constant;

  switch (pt_type) {
#define SWITCH_CASE_INT_AUX(_PtT_, _BuiltinT_, _Bits_, _Sign_)               \
  case _PtT_: {                                                              \
    auto element_type = mlir::IntegerType::get(                              \
        &mlir_ctx_, _Bits_, mlir::IntegerType::SignednessSemantics::_Sign_); \
                                                                             \
    auto ranked_type = mlir::RankedTensorType::get(shape, element_type);     \
    auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(ranked_type);        \
                                                                             \
    auto ptr = reinterpret_cast<const _BuiltinT_ *>(data);                   \
    auto array = llvm::ArrayRef<_BuiltinT_>(ptr, numel);                     \
                                                                             \
    constant = builder_.create<mlir::arith::ConstantOp>(                     \
        loc_, mlir::DenseElementsAttr::get(shaped_type, array));             \
                                                                             \
    break;                                                                   \
  }

#define SWITCH_CASE_INT(_PtT_, _BuiltinT_, _Sign_) \
  SWITCH_CASE_INT_AUX(_PtT_, _BuiltinT_, sizeof(_BuiltinT_) * 8, _Sign_)

    SWITCH_CASE_INT_AUX(PT_I1, bool, 1, Signless)
    SWITCH_CASE_INT(PT_I8, int8_t, Signless)
    SWITCH_CASE_INT(PT_U8, uint8_t, Unsigned)
    SWITCH_CASE_INT(PT_I16, int16_t, Signless)
    SWITCH_CASE_INT(PT_U16, uint16_t, Unsigned)
    SWITCH_CASE_INT(PT_I32, int32_t, Signless)
    SWITCH_CASE_INT(PT_U32, uint32_t, Unsigned)
    SWITCH_CASE_INT(PT_I64, int64_t, Signless)
    SWITCH_CASE_INT(PT_U64, uint64_t, Unsigned)
    SWITCH_CASE_INT(PT_I128, int128_t, Signless)

#undef SWITCH_CASE_INT
#undef SWITCH_CASE_INT_AUX

#define SWITCH_CASE_FLOAT(_PtT_, _BuiltinT_, _MlirT_)                    \
  case _PtT_: {                                                          \
    auto element_type = mlir::FloatType::get##_MlirT_(&mlir_ctx_);       \
    auto ranked_type = mlir::RankedTensorType::get(shape, element_type); \
    auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(ranked_type);    \
                                                                         \
    auto ptr = reinterpret_cast<const _BuiltinT_ *>(data);               \
    auto array = llvm::ArrayRef<_BuiltinT_>(ptr, numel);                 \
                                                                         \
    constant = builder_.create<mlir::arith::ConstantOp>(                 \
        loc_, mlir::DenseElementsAttr::get(shaped_type, array));         \
                                                                         \
    break;                                                               \
  }

    SWITCH_CASE_FLOAT(PT_F16, half_float::half, F16)
    SWITCH_CASE_FLOAT(PT_F32, float, F32)
    SWITCH_CASE_FLOAT(PT_F64, double, F64)

#undef SWITCH_CASE_FLOAT

    default: {
      SPU_ENFORCE(false, "should not be here");
      return {};
    }
  }

  if (out_shape != shape) {
    constant = Broadcast(constant, out_shape, {0});
  }

  return constant;
}

#define IMP_UNARY_OP(_Fun_, _Op_)                           \
  mlir::Value HloBuilder::_Fun_(const mlir::Value &input) { \
    return builder_.create<_Op_>(loc_, input);              \
  }

IMP_UNARY_OP(Not, NotOp)
IMP_UNARY_OP(Sine, SineOp)
IMP_UNARY_OP(Cosine, CosineOp)

#undef IMP_UNARY_OP

#define IMP_BINARY_OP(_Fun_, _Op_)                        \
  mlir::Value HloBuilder::_Fun_(const mlir::Value &lhs,   \
                                const mlir::Value &rhs) { \
    return builder_.create<_Op_>(loc_, lhs, rhs);         \
  }

IMP_BINARY_OP(Add, AddOp)
IMP_BINARY_OP(Sub, SubtractOp)
IMP_BINARY_OP(Mul, MulOp)
IMP_BINARY_OP(Div, DivOp)
IMP_BINARY_OP(Equal, EqualOp)
IMP_BINARY_OP(And, AndOp)
IMP_BINARY_OP(Xor, XorOp)
IMP_BINARY_OP(Or, OrOp)
IMP_BINARY_OP(NotEqual, NotEqualOp)
IMP_BINARY_OP(Max, MaxOp)
IMP_BINARY_OP(Min, MinOp)
IMP_BINARY_OP(Greater, GreaterOp)
IMP_BINARY_OP(GreaterEqual, GreaterEqualOp)
IMP_BINARY_OP(Less, LessOp)
IMP_BINARY_OP(LessEqual, LessEqualOp)
IMP_BINARY_OP(Remainder, RemOp)

#undef IMP_BINARY_OP

mlir::Value HloBuilder::Seal(const mlir::Value &input) {
  auto type = type_tools_.getType(input.getType(),
                                  mlir::spu::pphlo::Visibility::SECRET);
  return builder_.create<ConvertOp>(loc_, type, input);
}

mlir::Value HloBuilder::Reveal(const mlir::Value &input) {
  auto type = type_tools_.getType(input.getType(),
                                  mlir::spu::pphlo::Visibility::PUBLIC);
  return builder_.create<ConvertOp>(loc_, type, input);
}

mlir::Value HloBuilder::Cast(const mlir::Value &input,
                             spu::Visibility dst_vtype, spu::PtType dst_dtype) {
  auto visibility_type = GetVisibilityType(dst_vtype);
  auto element_type = GetElementType(&mlir_ctx_, dst_dtype);
  auto input_type = mlir::dyn_cast<mlir::ShapedType>(input.getType());
  SPU_ENFORCE(input_type);

  auto ranked_type =
      mlir::RankedTensorType::get(input_type.getShape(), element_type);
  auto type = type_tools_.getType(ranked_type, visibility_type);

  return builder_.create<ConvertOp>(loc_, type, input);
}

mlir::Value HloBuilder::Concatenate(const std::vector<mlir::Value> &ops,
                                    int64_t axis) {
  return builder_.create<ConcatenateOp>(loc_, ops, axis);
}

mlir::Value HloBuilder::Pad(const mlir::Value &input,
                            const mlir::Value &pad_value, const Sizes &edge_low,
                            const Sizes &edge_high, const Sizes &inner) {
  auto pv_type = mlir::dyn_cast<mlir::ShapedType>(pad_value.getType());
  SPU_ENFORCE(pv_type && pv_type.getShape().size() == 0);

  return builder_.create<PadOp>(loc_, input, pad_value, edge_low, edge_high,
                                inner);
}

mlir::Value HloBuilder::Reduce(absl::Span<const mlir::Value> inputs,
                               absl::Span<const mlir::Value> init_values,
                               const Axes &dims_to_reduce,
                               ReduceType reduce_type,
                               bool ignore_init_values) {
  SPU_ENFORCE(!inputs.empty());
  auto input_type = mlir::dyn_cast<mlir::ShapedType>(inputs[0].getType());

  SPU_ENFORCE(input_type);
  auto input_shape = input_type.getShape();

  Shape output_shape;
  Shape arg_shape;

  for (int64_t i = 0, ndims = input_shape.size(); i < ndims; ++i) {
    for (auto dim : dims_to_reduce) {
      SPU_ENFORCE(-ndims <= dim && dim < ndims);
      dim = (dim + ndims) % ndims;

      if (i == dim) {
        arg_shape.push_back(input_shape[i]);
      } else {
        output_shape.push_back(input_shape[i]);
      }
    }
  }

  auto element_type = input_type.getElementType();
  auto output_type = mlir::RankedTensorType::get(output_shape, element_type);
  auto arg_type = (!ignore_init_values && !init_values.empty())
                      ? init_values[0].getType()
                      : mlir::RankedTensorType::get(arg_shape, element_type);

  auto reduce = builder_.create<ReduceOp>(
      loc_, output_type, mlir::ValueRange(inputs.data(), inputs.size()),
      ignore_init_values
          ? mlir::ValueRange()
          : mlir::ValueRange(init_values.data(), init_values.size()),
      dims_to_reduce);

  auto &block = reduce.getBody().emplaceBlock();

  block.addArgument(arg_type, loc_);
  block.addArgument(arg_type, loc_);

  auto insert_point = builder_.saveInsertionPoint();
  builder_.setInsertionPointToStart(&block);

  mlir::Value out;
  switch (reduce_type) {
    case REDUCE_SUM: {
      out = Add(block.getArgument(0), block.getArgument(1));
      break;
    }
    case REDUCE_MAX: {
      out = Max(block.getArgument(0), block.getArgument(1));
      break;
    }
    case REDUCE_MIN: {
      out = Min(block.getArgument(0), block.getArgument(1));
      break;
    }
    default: {
      SPU_ENFORCE(false, "failed to support reduce type {}",
                  static_cast<uint32_t>(reduce_type));
      return {};
    }
  }

  builder_.create<ReturnOp>(loc_, mlir::ValueRange{out});
  builder_.restoreInsertionPoint(insert_point);

  return reduce.getResult(0);
}

mlir::Type HloBuilder::CommonType(llvm::ArrayRef<mlir::Type> types) {
  unsigned int float_width = 0;
  unsigned int signless_width = 0;
  unsigned int unsigned_width = 0;

  for (auto type : types) {
    llvm::TypeSwitch<mlir::Type, void>(type)
        .Case([&](mlir::IntegerType t) {
          if (t.isUnsigned()) {
            unsigned_width = std::max(unsigned_width, t.getWidth());
          } else {
            signless_width = std::max(signless_width, t.getWidth());
          }
        })
        .Case([&](mlir::FloatType t) {
          float_width = std::max(float_width, t.getWidth());
        })
        .Default([](mlir::Type) { SPU_THROW("Should not hit"); });
  }

  switch (float_width) {
    case 16:
      return builder_.getF16Type();
    case 32:
      return builder_.getF32Type();
    case 64:
      return builder_.getF64Type();
    default:
      // Nothing
      (void)float_width;
  }

  if (signless_width > unsigned_width) {
    return builder_.getIntegerType(signless_width);
  }

  SPU_ENFORCE(unsigned_width > 0);

  return builder_.getIntegerType(unsigned_width, false);
}

mlir::Value HloBuilder::Select(const mlir::Value &pred,
                               const mlir::Value &on_true,
                               const mlir::Value &on_false) {
  auto on_true_base = type_tools_.getBaseType(on_true.getType());
  auto on_false_base = type_tools_.getBaseType(on_false.getType());

  auto ret_vis = type_tools_.computeCommonVisibility(
      {type_tools_.getTypeVisibility(pred.getType()),
       type_tools_.getTypeVisibility(on_true.getType()),
       type_tools_.getTypeVisibility(on_false.getType())});

  if (on_true_base == on_false_base) {
    auto ret_type = type_tools_.getType(on_true.getType(), ret_vis);
    return builder_.create<SelectOp>(loc_, ret_type, pred, on_true, on_false);
  }

  auto common_type = CommonType({on_true_base, on_false_base});

  auto casted_on_true = builder_.create<ConvertOp>(
      loc_, type_tools_.replaceBaseType(on_true.getType(), common_type),
      on_true);
  auto casted_on_false = builder_.create<ConvertOp>(
      loc_, type_tools_.replaceBaseType(on_false.getType(), common_type),
      on_false);

  auto ret_type = type_tools_.getType(casted_on_true.getType(), ret_vis);

  return builder_.create<SelectOp>(loc_, ret_type, pred, casted_on_true,
                                   casted_on_false);
}

std::vector<mlir::Value> HloBuilder::SimpleSort(
    absl::Span<const mlir::Value> inputs, int64_t sort_dim,
    SortDirection direction, int64_t num_keys) {
  llvm::SmallVector<mlir::Type> result_types;
  result_types.reserve(inputs.size());

  for (const auto &input : inputs) {
    result_types.emplace_back(input.getType());
  }

  auto op = builder_.create<SimpleSortOp>(
      loc_, result_types, mlir::ValueRange(inputs.data(), inputs.size()),
      sort_dim, num_keys, direction);

  auto results = op.getResults();
  return {results.begin(), results.end()};
}

mlir::Value HloBuilder::Slice(const mlir::Value &input, const Index &start,
                              const Index &end, const Strides &strides) {
  return builder_.create<SliceOp>(loc_, input, start, end, strides);
}

void HloBuilder::Return(const std::vector<mlir::Value> &outputs) {
  auto arg_type_list = main_fun_op_.getArgumentTypes();

  auto ret_type_list =
      llvm::map_to_vector(outputs, [](mlir::Value v) { return v.getType(); });

  main_fun_op_.setFunctionType(
      builder_.getFunctionType(arg_type_list, ret_type_list));

  builder_.create<mlir::func::ReturnOp>(loc_, outputs);
}

std::vector<mlir::Value> HloBuilder::Shuffle(
    absl::Span<const mlir::Value> inputs, int64_t axis) {
  SPU_ENFORCE(!inputs.empty());

  llvm::SmallVector<mlir::Type> result_types;
  result_types.reserve(inputs.size());

  for (const auto &input : inputs) {
    result_types.emplace_back(input.getType());
  }

  auto op = builder_.create<CustomCallOp>(
      loc_, result_types, mlir::ValueRange(inputs.data(), inputs.size()),
      "hlo.shuffle", true);

  auto attr = mlir::DictionaryAttr::get(
      &mlir_ctx_, {mlir::NamedAttribute(builder_.getStringAttr("axis"),
                                        builder_.getI64IntegerAttr(axis))});
  op->setAttr("mhlo.attributes", attr);

  auto results = op.getResults();
  return {results.begin(), results.end()};
}

mlir::Value HloBuilder::FilterByMask(const mlir::Value &input,
                                     absl::Span<const uint8_t> mask) {
  auto input_type = mlir::dyn_cast<mlir::ShapedType>(input.getType());
  SPU_ENFORCE(input_type);

  int64_t numel = std::accumulate(mask.begin(), mask.end(), 0);
  auto output_type =
      mlir::RankedTensorType::get({numel}, input_type.getElementType());

  auto op = builder_.create<CustomCallOp>(loc_, output_type, input,
                                          "hlo.filter_by_mask", true);

  auto attr = mlir::DictionaryAttr::get(
      &mlir_ctx_,
      {mlir::NamedAttribute(
          builder_.getStringAttr("mask"),
          builder_.getDenseI8ArrayAttr(llvm::ArrayRef<int8_t>(
              reinterpret_cast<const int8_t *>(mask.data()), mask.size())))});
  op->setAttr("mhlo.attributes", attr);

  return op.getResult(0);
}

mlir::Value HloBuilder::LinearGather(const mlir::Value &input,
                                     const Index &indices) {
  auto input_type = mlir::dyn_cast<mlir::ShapedType>(input.getType());
  SPU_ENFORCE(input_type);

  // follow NdArrayRef::linear_gather
  auto output_type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(indices.size())}, input_type.getElementType());

  auto op = builder_.create<CustomCallOp>(loc_, output_type, input,
                                          "hlo.linear_gather", true);

  auto attr = mlir::DictionaryAttr::get(
      &mlir_ctx_,
      {mlir::NamedAttribute(builder_.getStringAttr("indices"),
                            builder_.getDenseI64ArrayAttr(indices))});
  op->setAttr("mhlo.attributes", attr);

  return op.getResult(0);
}

mlir::Value HloBuilder::LinearScatter(const mlir::Value &input,
                                      const mlir::Value &update,
                                      const Index &indices) {
  llvm::SmallVector<mlir::Value> inputs = {input, update};
  auto op = builder_.create<CustomCallOp>(
      loc_, input.getType(), mlir::ValueRange(inputs.data(), inputs.size()),
      "hlo.linear_scatter", true);

  auto attr = mlir::DictionaryAttr::get(
      &mlir_ctx_,
      {mlir::NamedAttribute(builder_.getStringAttr("indices"),
                            builder_.getDenseI64ArrayAttr(indices))});
  op->setAttr("mhlo.attributes", attr);

  return op.getResult(0);
}

mlir::Value HloBuilder::Broadcast(const mlir::Value &input,
                                  const Shape &to_shape, const Axes &in_dims) {
  auto input_type = mlir::dyn_cast<mlir::ShapedType>(input.getType());
  SPU_ENFORCE(input_type);

  auto output_type =
      mlir::RankedTensorType::get(to_shape, input_type.getElementType());

  return builder_.create<BroadcastOp>(loc_, output_type, input, in_dims);
}

std::shared_ptr<yacl::Buffer> DumpSecret(SPUContext *ctx,
                                         const spu::MemRef &input,
                                         spu::PtType pt_type,
                                         int64_t fxp_bits) {
  SPU_ENFORCE(!input.isPublic());
  auto value = hal::reveal(ctx, input);

  auto buffer = std::make_shared<yacl::Buffer>(SizeOf(pt_type) * value.numel());
  PtBufferView buffer_view(buffer->data(), pt_type, value.shape(), {});

  if (pt_type == PT_F16 || pt_type == PT_F32 || pt_type == PT_F64) {
    if (fxp_bits <= 0) {
      fxp_bits = ctx->config().fxp_fraction_bits();
    }

    hal::_decode_fp(ctx, value, &buffer_view, fxp_bits);
  } else {
    hal::_decode_int(ctx, value, &buffer_view);
  }

  return buffer;
}

std::shared_ptr<yacl::Buffer> HloBuilder::Dump(SPUContext *ctx,
                                               const spu::MemRef &input,
                                               spu::PtType pt_type,
                                               int64_t fxp_bits) {
  SPU_ENFORCE(pt_type != PT_INVALID);

  if (input.isPublic()) {
    // public data is not encoded
    SPU_ENFORCE(SizeOf(pt_type) == input.elsize());
    return input.clone().buf();
  } else {
    return DumpSecret(ctx, input, pt_type, fxp_bits);
  }
}

}  // namespace spu::kernel::hlo
