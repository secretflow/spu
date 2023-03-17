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

#include "libspu/dialect/pphlo_ops.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/utils.h"

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
    case DT_FXP:
      return mlir::Float32Type::get(mlir_ctx);
    default:
      SPU_THROW("Should not hit");
  }
}

mlir::TensorType buildMLIRType(mlir::MLIRContext *mlir_ctx,
                               const spu::Value &v) {
  return mlir::RankedTensorType::get(v.shape(), getElementType(mlir_ctx, v));
}

mlir::stablehlo::Tensor convertToStablehloTensor(mlir::MLIRContext *mlir_ctx,
                                                 HalContext *ctx,
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
      buf.emplace_back(*reinterpret_cast<const float *>(iter.getRawPtr()));
    }
    return mlir::stablehlo::makeTensor(
        mlir::DenseElementsAttr::get(shaped_type, buf));
  }

  llvm::SmallVector<llvm::APInt> buf;

  if (v.dtype() == DT_I1) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const bool *>(iter.getRawPtr());
      buf.emplace_back(1, v);
    }
  }

  if (v.dtype() == DT_I8) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int8_t *>(iter.getRawPtr());
      buf.emplace_back(8, v, true);
    }
  }

  if (v.dtype() == DT_U8) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint8_t *>(iter.getRawPtr());
      buf.emplace_back(8, v);
    }
  }

  if (v.dtype() == DT_I16) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int16_t *>(iter.getRawPtr());
      buf.emplace_back(16, v, true);
    }
  }

  if (v.dtype() == DT_U16) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint16_t *>(iter.getRawPtr());
      buf.emplace_back(16, v);
    }
  }

  if (v.dtype() == DT_I32) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int32_t *>(iter.getRawPtr());
      buf.emplace_back(32, v, true);
    }
  }

  if (v.dtype() == DT_U32) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint32_t *>(iter.getRawPtr());
      buf.emplace_back(32, v);
    }
  }

  if (v.dtype() == DT_I64) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const int64_t *>(iter.getRawPtr());
      buf.emplace_back(64, v, true);
    }
  }

  if (v.dtype() == DT_U64) {
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter) {
      uint64_t v = *reinterpret_cast<const uint64_t *>(iter.getRawPtr());
      buf.emplace_back(64, v);
    }
  }

  return mlir::stablehlo::makeTensor(
      mlir::DenseElementsAttr::get(shaped_type, buf));
}

bool verifyScalar(const mlir::stablehlo::Element &hlo_e,
                  const std::byte *expected) {
  // return false;
  auto t = hlo_e.getType();
  if (t.isF32()) {
    auto hlo_f = hlo_e.getFloatValue().convertToFloat();
    auto exp_f = *reinterpret_cast<const float *>(expected);
    return std::abs(hlo_f - exp_f) <= 1e-2;
  }

  if (t.isInteger(1)) {
    return hlo_e.getBooleanValue() == *reinterpret_cast<const bool *>(expected);
  }

  auto hlo_i = hlo_e.getIntegerValue().getLimitedValue();
  if (t.isSignlessInteger(8)) {
    return static_cast<int8_t>(hlo_i) ==
           *reinterpret_cast<const int8_t *>(expected);
  }

  if (t.isUnsignedInteger(8)) {
    return static_cast<uint8_t>(hlo_i) ==
           *reinterpret_cast<const uint8_t *>(expected);
  }

  if (t.isSignlessInteger(16)) {
    return static_cast<int16_t>(hlo_i) ==
           *reinterpret_cast<const int16_t *>(expected);
  }

  if (t.isUnsignedInteger(16)) {
    return static_cast<uint16_t>(hlo_i) ==
           *reinterpret_cast<const uint16_t *>(expected);
  }

  if (t.isSignlessInteger(32)) {
    return static_cast<int32_t>(hlo_i) ==
           *reinterpret_cast<const int32_t *>(expected);
  }

  if (t.isUnsignedInteger(32)) {
    return static_cast<uint32_t>(hlo_i) ==
           *reinterpret_cast<const uint32_t *>(expected);
  }

  if (t.isSignlessInteger(64)) {
    return static_cast<int64_t>(hlo_i) ==
           *reinterpret_cast<const int64_t *>(expected);
  }

  if (t.isUnsignedInteger(64)) {
    return static_cast<uint64_t>(hlo_i) ==
           *reinterpret_cast<const uint64_t *>(expected);
  }

  return false;
}

bool verifyEqual(const mlir::stablehlo::Tensor &hlo_ret,
                 const NdArrayRef &expected) {
  bool pass = true;
  auto numel = expected.numel();
  size_t mismatch = 0;

  spu::kernel::forEachIndex(
      expected.shape(), [&](absl::Span<const int64_t> output_index) {
        mlir::stablehlo::Index idx{
            llvm::ArrayRef<int64_t>(output_index.begin(), output_index.end())};
        auto xla_value = hlo_ret.get(idx);
        const auto *spu_value = &expected.at(output_index);
        bool equal = verifyScalar(xla_value, spu_value);
        if (!equal) {
          // SPDLOG_INFO(
          //     "Equal check failed at ({}), xla_value = {}, spu_value= {}",
          //     fmt::join(output_index, ","), xla_value, spu_value);
          pass = false;
          ++mismatch;
        }
      });

  SPDLOG_INFO("Answer has {} elements, {} mismatch found", numel, mismatch);
  return pass;
#undef CASE
}

bool verifyEqual(HalContext *ctx, const mlir::stablehlo::Tensor &ret,
                 const spu::Value &expected) {
  NdArrayRef arr;
  if (expected.isSecret()) {
    arr = kernel::hal::dump_public(ctx, kernel::hal::reveal(ctx, expected));
  } else {
    arr = kernel::hal::dump_public(ctx, expected);
  }
  return verifyEqual(ret, arr);
}

}  // namespace

PPHloVerifier::PPHloVerifier(HalContext *ctx) : ctx_(ctx) {
  mlir_ctx_.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
}

void PPHloVerifier::verify(mlir::pphlo::AbsOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalAbsOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::ReciprocalOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::NegOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalNegOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::LogOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalLogOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::Log1pOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::FloorOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalFloorOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::CeilOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalCeilOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::LogisticOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::TanhOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalTanhOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::NotOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalNotOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::ExpOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalExponentialOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::Expm1Op op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::RsqrtOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalRsqrtOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::SqrtOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto ret = mlir::stablehlo::evalSqrtOp(t1, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::SignOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::RoundOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::AddOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalAddOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::SubtractOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalSubtractOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::MulOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalMultiplyOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::PowOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalPowerOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::MaxOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalMaxOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::MinOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalMinOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::AndOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalAndOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::OrOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalOrOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::XorOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalXorOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::DivOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalDivideOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::RemOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto ret = mlir::stablehlo::evalRemOp(t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::DotOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::DotGeneralOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::EqualOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::NotEqualOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::LessOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::LessEqualOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::GreaterOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::GreaterEqualOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::SelectOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto pred = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[2]);
  auto ret = mlir::stablehlo::evalSelectOp(pred, t1, t2, t1.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::ClampOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto min_ = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto op_ = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  auto max_ = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[2]);
  auto ret = mlir::stablehlo::evalClampOp(min_, op_, max_, op_.getType());
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::BitcastConvertOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::ConvertOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::ConvolutionOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::DynamicSliceOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  llvm::SmallVector<mlir::stablehlo::Tensor> start_indices;
  for (size_t idx = 1; idx < operands.size(); ++idx) {
    SPDLOG_INFO(operands[idx]);
    start_indices.emplace_back(
        convertToStablehloTensor(&mlir_ctx_, ctx_, operands[idx]));
  }
  auto ret = mlir::stablehlo::evalDynamicSliceOp(
      t1, start_indices, mlir::stablehlo::Sizes(op.getSliceSizes()),
      buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::DynamicUpdateSliceOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto update = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);
  llvm::SmallVector<mlir::stablehlo::Tensor> start_indices;
  for (size_t idx = 2; idx < operands.size(); ++idx) {
    start_indices.emplace_back(
        convertToStablehloTensor(&mlir_ctx_, ctx_, operands[idx]));
  }
  auto ret = mlir::stablehlo::evalDynamicUpdateSliceOp(
      t1, update, start_indices, buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::GatherOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::PadOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);
  auto t2 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[1]);

  auto ret = mlir::stablehlo::evalPadOp(
      t1, t2, mlir::stablehlo::Sizes(op.getEdgePaddingLow()),
      mlir::stablehlo::Sizes(op.getInteriorPadding()),
      buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::BroadcastOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);

  auto ret = mlir::stablehlo::evalBroadcastInDimOp(
      t1, mlir::stablehlo::Axes(op.getBroadcastDimensionsAttr()),
      buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::ConcatenateOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  llvm::SmallVector<mlir::stablehlo::Tensor> vals;
  for (const auto &operand : operands) {
    vals.emplace_back(convertToStablehloTensor(&mlir_ctx_, ctx_, operand));
  }
  auto ret = mlir::stablehlo::evalConcatenateOp(
      vals, op.getDimension(), buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::ReshapeOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);

  auto ret = mlir::stablehlo::evalReshapeOp(
      t1, buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::ReverseOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);

  auto ret = mlir::stablehlo::evalReverseOp(
      t1, mlir::stablehlo::Axes(op.getDimensions()),
      buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::SliceOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);

  auto ret = mlir::stablehlo::evalSliceOp(
      t1, mlir::stablehlo::Index(op.getStartIndices()),
      mlir::stablehlo::Sizes(op.getStrides()),
      buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::TransposeOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto t1 = convertToStablehloTensor(&mlir_ctx_, ctx_, operands[0]);

  auto ret = mlir::stablehlo::evalTransposeOp(
      t1, mlir::stablehlo::Axes(op.getPermutation()),
      buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::IotaOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  auto ret = mlir::stablehlo::evalIotaOp(
      op.getIotaDimension(), buildMLIRType(&mlir_ctx_, expected[0]));
  mismatch_handler_(verifyEqual(ctx_, ret, expected[0]));
}

void PPHloVerifier::verify(mlir::pphlo::ReduceOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::ReduceWindowOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::SelectAndScatterOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::ShiftLeftOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::ShiftRightArithmeticOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::ShiftRightLogicalOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

void PPHloVerifier::verify(mlir::pphlo::SortOp op,
                           absl::Span<const spu::Value> operands,
                           absl::Span<const spu::Value> expected) {
  SPU_THROW("Missing stablehlo interpreter support");
}

}  // namespace spu::device::pphlo
