// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/device/pphlo/pphlo_verifier.h"

#include <utility>

#include "spdlog/spdlog.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Tensor.h"

#include "libspu/dialect/pphlo/ops.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::device::pphlo {
namespace {

mlir::Type getElementType(mlir::MLIRContext *mlir_ctx, const spu::Value &v) {
  switch (v.dtype()) {
    case DT_I1:
      return mlir::IntegerType::get(mlir_ctx, 1);
    case DT_I8:
      return mlir::IntegerType::get(mlir_ctx, 8);
    case DT_U8:
      return mlir::IntegerType::get(mlir_ctx, 8, mlir::IntegerType::Unsigned);
    case DT_I16:
      return mlir::IntegerType::get(mlir_ctx, 16);
    case DT_U16:
      return mlir::IntegerType::get(mlir_ctx, 16, mlir::IntegerType::Unsigned);
    case DT_I32:
      return mlir::IntegerType::get(mlir_ctx, 32);
    case DT_U32:
      return mlir::IntegerType::get(mlir_ctx, 32, mlir::IntegerType::Unsigned);
    case DT_I64:
      return mlir::IntegerType::get(mlir_ctx, 64);
    case DT_U64:
      return mlir::IntegerType::get(mlir_ctx, 64, mlir::IntegerType::Unsigned);
    case DT_F16:
      return mlir::Float16Type::get(mlir_ctx);
    case DT_F32:
      return mlir::Float32Type::get(mlir_ctx);
    case DT_F64:
      return mlir::Float64Type::get(mlir_ctx);
    default:
      SPU_THROW("Should not hit");
  }
}

mlir::TensorType buildMLIRType(mlir::MLIRContext *mlir_ctx,
                               const spu::Value &v) {
  return mlir::RankedTensorType::get(v.shape(), getElementType(mlir_ctx, v));
}

mlir::stablehlo::Tensor convertToStablehloTensor(mlir::MLIRContext *mlir_ctx,
                                                 SPUContext *ctx,
                                                 const spu::Value &v) {
  NdArrayRef arr;
  if (v.isSecret()) {
    arr = kernel::hal::dump_public(ctx, kernel::hal::reveal(ctx, v));
  } else {
    arr = kernel::hal::dump_public(ctx, v);
  }

  auto shaped_type = buildMLIRType(mlir_ctx, v);

  if (v.isFxp()) {
    llvm::SmallVector<llvm::APFloat> buf;
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      buf.emplace_back(*reinterpret_cast<const float *>(&*iter));
    }
    return mlir::stablehlo::makeTensor(
        mlir::DenseElementsAttr::get(shaped_type, buf));
  }

  llvm::SmallVector<llvm::APInt> buf;

  if (v.dtype() == DT_I1) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const bool *>(&*iter);
      buf.emplace_back(1, v);
    }
  }

  if (v.dtype() == DT_I8) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int8_t *>(&*iter);
      buf.emplace_back(8, v, true);
    }
  }

  if (v.dtype() == DT_U8) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint8_t *>(&*iter);
      buf.emplace_back(8, v);
    }
  }

  if (v.dtype() == DT_I16) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int16_t *>(&*iter);
      buf.emplace_back(16, v, true);
    }
  }

  if (v.dtype() == DT_U16) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint16_t *>(&*iter);
      buf.emplace_back(16, v);
    }
  }

  if (v.dtype() == DT_I32) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int32_t *>(&*iter);
      buf.emplace_back(32, v, true);
    }
  }

  if (v.dtype() == DT_U32) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint32_t *>(&*iter);
      buf.emplace_back(32, v);
    }
  }

  if (v.dtype() == DT_I64) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int64_t *>(&*iter);
      buf.emplace_back(64, v, true);
    }
  }

  if (v.dtype() == DT_U64) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint64_t *>(&*iter);
      buf.emplace_back(64, v);
    }
  }

  return mlir::stablehlo::makeTensor(
      mlir::DenseElementsAttr::get(shaped_type, buf));
}

#define PRINT_DETAIL_LIMIT 10
bool verifyScalar(const mlir::stablehlo::Element &xla_el,
                  const mlir::stablehlo::Element &spu_el,
                  int64_t mismatch_counter) {
  // return false;
  auto t = xla_el.getType();
  if (t.isF32()) {
    auto xla_f = xla_el.getFloatValue().convertToFloat();
    auto spu_f = spu_el.getFloatValue().convertToFloat();
    bool ret = std::abs(xla_f - spu_f) <= 1e-2;
    if (!ret && mismatch_counter < PRINT_DETAIL_LIMIT) {
      SPDLOG_INFO("fxp mismatch xla_value = {}, spu_value = {}", xla_f, spu_f);
    }
    return ret;
  }

  if (t.isInteger(1)) {
    auto ret = xla_el.getBooleanValue() == spu_el.getBooleanValue();
    if (!ret && mismatch_counter < PRINT_DETAIL_LIMIT) {
      SPDLOG_INFO("boolean mismatch xla_value = {}, spu_value = {}",
                  xla_el.getBooleanValue(), spu_el.getBooleanValue());
    }
    return ret;
  }

  auto ret = xla_el.getIntegerValue() == spu_el.getIntegerValue();
  if (!ret && mismatch_counter < PRINT_DETAIL_LIMIT) {
    SPDLOG_INFO("int mismatch xla_value = {}, spu_value = {}",
                xla_el.getIntegerValue().getLimitedValue(),
                spu_el.getIntegerValue().getLimitedValue());
  }
  return ret;
}

std::string mlirTypeToString(mlir::Type type) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  type.print(os);
  return os.str();
}

bool verifyEqual(SPUContext *, const mlir::stablehlo::Tensor &xla_ret,
                 const mlir::stablehlo::Tensor &spu_ret) {
  if (xla_ret.getType() != spu_ret.getType()) {
    SPDLOG_INFO("Answer has a type mismatch, xla type = {}, spu type = {}",
                mlirTypeToString(xla_ret.getType()),
                mlirTypeToString(spu_ret.getType()));
    return false;
  }

  if (xla_ret.getNumElements() != spu_ret.getNumElements()) {
    SPDLOG_ERROR("Number of element mismatch, xla numel = {}, spu numel = {}",
                 xla_ret.getNumElements(), spu_ret.getNumElements());
    return false;
  }

  if (xla_ret.getShape() != spu_ret.getShape()) {
    SPDLOG_ERROR("Number of element mismatch, xla numel = {}, spu numel = {}",
                 fmt::join(xla_ret.getShape(), "x"),
                 fmt::join(spu_ret.getShape(), "x"));
    return false;
  }

  bool pass = true;
  auto numel = xla_ret.getNumElements();
  int64_t mismatch = 0;

  auto iter = spu_ret.index_begin();
  auto iter_end = spu_ret.index_end();

  while (iter != iter_end) {
    bool equal = verifyScalar(xla_ret.get(*iter), spu_ret.get(*iter), mismatch);
    if (!equal) {
      pass = false;
      ++mismatch;
    }
    ++iter;
  }

  if (!pass) {
    SPDLOG_INFO("Answer has {} elements, {} mismatch found", numel, mismatch);
  }
  return pass;
}

}  // namespace

PPHloVerifier::PPHloVerifier(SPUContext *ctx) : ctx_(ctx) {
  mlir_ctx_.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
}

#define UNARY_VERIFIER(OP_TYPE, FCN_NAME)                                   \
  void PPHloVerifier::verify(mlir::spu::pphlo::OP_TYPE,                     \
                             absl::Span<const spu::Value> operands,         \
                             absl::Span<const spu::Value> expected) {       \
    auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);      \
    auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]); \
    auto xla_ret = mlir::stablehlo::FCN_NAME(t1, t1.getType());             \
    mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));                 \
  }

UNARY_VERIFIER(AbsOp, absOp)
UNARY_VERIFIER(NegOp, negOp)
UNARY_VERIFIER(LogOp, logOp)
UNARY_VERIFIER(FloorOp, floorOp)
UNARY_VERIFIER(CeilOp, ceilOp)
UNARY_VERIFIER(LogisticOp, logisticOp)
UNARY_VERIFIER(TanhOp, tanhOp)
UNARY_VERIFIER(SineOp, sineOp)
UNARY_VERIFIER(CosineOp, cosineOp)
UNARY_VERIFIER(NotOp, notOp)
UNARY_VERIFIER(ExpOp, exponentialOp)
UNARY_VERIFIER(RsqrtOp, rsqrtOp)
UNARY_VERIFIER(SqrtOp, sqrtOp)
UNARY_VERIFIER(RoundOp, roundOp)
UNARY_VERIFIER(RoundNearestEvenOp, roundNearestEvenOp)
UNARY_VERIFIER(SignOp, signOp)
UNARY_VERIFIER(Log1pOp, log1pOp)
UNARY_VERIFIER(Expm1Op, expm1Op)

#undef UNARY_VERIFIER

#define BINARY_VERIFIER(OP_TYPE, FCN_NAME)                                  \
  void PPHloVerifier::verify(mlir::spu::pphlo::OP_TYPE,                     \
                             absl::Span<const spu::Value> operands,         \
                             absl::Span<const spu::Value> expected) {       \
    auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);      \
    auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);      \
    auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]); \
    auto xla_ret = mlir::stablehlo::FCN_NAME(t1, t2, t1.getType());         \
    mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));                 \
  }

BINARY_VERIFIER(AddOp, addOp)
BINARY_VERIFIER(Atan2Op, atan2Op)
BINARY_VERIFIER(SubtractOp, subtractOp)
BINARY_VERIFIER(MulOp, multiplyOp)
BINARY_VERIFIER(PowOp, powerOp)
BINARY_VERIFIER(MaxOp, maxOp)
BINARY_VERIFIER(MinOp, minOp)
BINARY_VERIFIER(AndOp, andOp)
BINARY_VERIFIER(OrOp, orOp)
BINARY_VERIFIER(XorOp, xorOp)
BINARY_VERIFIER(DivOp, divideOp)
BINARY_VERIFIER(RemOp, remOp)
BINARY_VERIFIER(ShiftLeftOp, shiftLeftOp)
BINARY_VERIFIER(ShiftRightLogicalOp, shiftRightLogicalOp)
BINARY_VERIFIER(ShiftRightArithmeticOp, shiftRightArithmeticOp)

#undef BINARY_VERIFIER

#define COMPARISON_VERIFIER(COMP_OP, KIND)                                  \
  void PPHloVerifier::verify(mlir::spu::pphlo::COMP_OP,                     \
                             absl::Span<const spu::Value> operands,         \
                             absl::Span<const spu::Value> expected) {       \
    auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);      \
    auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);      \
    auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]); \
    auto ret_type = mlir::RankedTensorType::get(                            \
        t1.getShape(), mlir::IntegerType::get(&mlir_ctx_, 1));              \
    auto xla_ret = mlir::stablehlo::compareOp(                              \
        t1, t2, mlir::stablehlo::ComparisonDirection::KIND, ret_type);      \
    mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));                 \
  }

COMPARISON_VERIFIER(EqualOp, EQ)
COMPARISON_VERIFIER(NotEqualOp, NE)
COMPARISON_VERIFIER(LessOp, LT)
COMPARISON_VERIFIER(LessEqualOp, LE)
COMPARISON_VERIFIER(GreaterOp, GT)
COMPARISON_VERIFIER(GreaterEqualOp, GE)

#undef COMPARISON_VERIFIER

void PPHloVerifier::verify(mlir::spu::pphlo::SelectOp,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto pred = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[2]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret = mlir::stablehlo::selectOp(pred, t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::ClampOp,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto min_ = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto op_ = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto max_ = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[2]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret = mlir::stablehlo::clampOp(min_, op_, max_, op_.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::DynamicSliceOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  llvm::SmallVector<mlir::stablehlo::Tensor> start_indices;
  for (size_t idx = 1; idx < operands.size(); ++idx) {
    start_indices.emplace_back(
        convertToStablehloTensor(&mlir_ctx_, ctx_, operands[idx]));
  }
  auto xla_ret = mlir::stablehlo::dynamicSliceOp(
      t1, start_indices, mlir::stablehlo::Sizes(op.getSliceSizes()),
      spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::DynamicUpdateSliceOp,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto update = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  llvm::SmallVector<mlir::stablehlo::Tensor> start_indices;
  for (size_t idx = 2; idx < operands.size(); ++idx) {
    start_indices.emplace_back(
        convertToStablehloTensor(&mlir_ctx_, ctx_, operands[idx]));
  }
  auto xla_ret = mlir::stablehlo::dynamicUpdateSliceOp(
      t1, update, start_indices, spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::PadOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);

  auto xla_ret = mlir::stablehlo::padOp(
      t1, t2, mlir::stablehlo::Sizes(op.getEdgePaddingLow()),
      mlir::stablehlo::Sizes(op.getInteriorPadding()), spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::BroadcastOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);

  auto xla_ret = mlir::stablehlo::broadcastInDimOp(
      t1, mlir::stablehlo::Axes(op.getBroadcastDimensionsAttr()),
      spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::ConcatenateOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  llvm::SmallVector<mlir::stablehlo::Tensor> vals;
  for (const auto &operand : operands) {
    vals.emplace_back(convertToStablehloTensor(&mlir_ctx_, ctx_, operand));
  }
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret = mlir::stablehlo::concatenateOp(vals, op.getDimension(),
                                                spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::ReshapeOp,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret = mlir::stablehlo::reshapeOp(t1, spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::ReverseOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret = mlir::stablehlo::reverseOp(
      t1, mlir::stablehlo::Axes(op.getDimensions()), spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::SliceOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret = mlir::stablehlo::sliceOp(
      t1, mlir::stablehlo::Index(op.getStartIndices()),
      mlir::stablehlo::Sizes(op.getStrides()), spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::TransposeOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret = mlir::stablehlo::transposeOp(
      t1, mlir::stablehlo::Axes(op.getPermutation()), spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::IotaOp op,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> expected) {
  auto spu_ret = convertToStablehloTensor(&mlir_ctx_, ctx_, expected[0]);
  auto xla_ret =
      mlir::stablehlo::iotaOp(op.getIotaDimension(), spu_ret.getType());
  mismatch_handler_(verifyEqual(ctx_, xla_ret, spu_ret));
}

void PPHloVerifier::verify(mlir::spu::pphlo::ReduceOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::ReduceWindowOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::SelectAndScatterOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::SortOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::BitcastConvertOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::ConvertOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_INFO("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::ConvolutionOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::ReciprocalOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::DotOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::spu::pphlo::DotGeneralOp,
                           absl::Span<const spu::Value> /*operands*/,
                           absl::Span<const spu::Value> /*expected*/) {
  SPDLOG_WARN("Missing stablehlo interpreter support");
}

}  // namespace spu::device::pphlo
