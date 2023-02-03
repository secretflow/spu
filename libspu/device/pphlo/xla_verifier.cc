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

#include "libspu/device/pphlo/xla_verifier.h"

#include "spdlog/spdlog.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/literal.h"
#include "xla/literal_util.h"

#include "libspu/dialect/pphlo_ops.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/utils.h"

namespace spu::device::pphlo {
namespace {

std::vector<int64_t> convertDenseIntElementAttr(
    const mlir::DenseIntElementsAttr &attr) {
  std::vector<int64_t> ret;

  for (const auto &v : attr.getValues<int64_t>()) {
    ret.emplace_back(v);
  }

  return ret;
}

xla::PrimitiveType getXlaType(PtTy type) {
  switch (type.pt_type()) {
    case spu::PtType::PT_BOOL:
      return xla::PRED;
    case spu::PtType::PT_I8:
      return xla::S8;
    case spu::PtType::PT_U8:
      return xla::U8;
    case spu::PtType::PT_I16:
      return xla::S16;
    case spu::PtType::PT_U16:
      return xla::U16;
    case spu::PtType::PT_I32:
      return xla::S32;
    case spu::PtType::PT_U32:
      return xla::U32;
    case spu::PtType::PT_I64:
      return xla::S64;
    case spu::PtType::PT_U64:
      return xla::U64;
    case spu::PtType::PT_F32:
      return xla::F32;
    case spu::PtType::PT_F64:
      return xla::F64;
    default:
      YACL_THROW("Unhandled type {}", type.toString());
  }
  return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

xla::Shape buildXLAShape(PtTy type, absl::Span<const int64_t> shape) {
  return xla::ShapeUtil::MakeShape(getXlaType(type), shape);
}  // namespace

#define ALL_POSSIBLE_PTTYPES(FN) \
  FN(PT_I8, int8_t, I8)          \
  FN(PT_U8, uint8_t, U8)         \
  FN(PT_I16, int16_t, I16)       \
  FN(PT_U16, uint16_t, U16)      \
  FN(PT_I32, int32_t, I32)       \
  FN(PT_U32, uint32_t, U32)      \
  FN(PT_I64, int64_t, I64)       \
  FN(PT_U64, uint64_t, U64)      \
  FN(PT_BOOL, bool, I1)          \
  FN(PT_F32, float, F32)         \
  FN(PT_F64, double, F64)

xla::Literal convertToXlaLiteral(HalContext *ctx, const spu::Value &v) {
  auto arr = kernel::hal::dump_public(ctx, v);
  auto xla_shape = buildXLAShape(*arr.eltype().as<PtTy>(), arr.shape());
  xla::Literal ret = xla::Literal::CreateFromShape(xla_shape);

#define CASE(NAME, TYPE, _)                                        \
  case NAME: {                                                     \
    spu::kernel::forEachIndex(                                     \
        arr.shape(), [&](absl::Span<const int64_t> output_index) { \
          ret.Set(output_index, arr.at<TYPE>(output_index));       \
        });                                                        \
    break;                                                         \
  }

  switch (arr.eltype().as<PtTy>()->pt_type()) {
    ALL_POSSIBLE_PTTYPES(CASE)

    default:
      YACL_THROW("unexpected type={}", arr.eltype());
  }

#undef CASE
  return ret;
}  // namespace

template <typename T>
xla::Literal createConstXlaLiteral(T v, const xla::Shape &shape) {
  auto scalar = xla::LiteralUtil::CreateR0<T>(v);
  return scalar.Broadcast(shape, {}).value();
}

xla::Literal xlaOnes(HalContext *ctx, const spu::Value &base) {
  auto arr = kernel::hal::dump_public(ctx, base);
#define CASE(NAME, TYPE, _)                                             \
  case NAME: {                                                          \
    return createConstXlaLiteral(                                       \
        TYPE(1), buildXLAShape(*arr.eltype().as<PtTy>(), arr.shape())); \
  }

  switch (arr.eltype().as<PtTy>()->pt_type()) {
    ALL_POSSIBLE_PTTYPES(CASE)

    default:
      YACL_THROW("unexpected type={}", arr.eltype());
  }

#undef CASE
}

bool verifyEqual(const xla::Literal &xla_ret, const NdArrayRef &expected) {
  bool pass = true;
  auto numel = expected.numel();
  size_t mismatch = 0;
#define CASE(NAME, TYPE, _)                                                  \
  case NAME: {                                                               \
    spu::kernel::forEachIndex(                                               \
        expected.shape(), [&](absl::Span<const int64_t> output_index) {      \
          auto xla_value = xla_ret.Get<TYPE>(output_index);                  \
          auto spu_value = expected.at<TYPE>(output_index);                  \
          bool equal = false;                                                \
          if constexpr (std::is_integral_v<TYPE>) {                          \
            equal = (xla_value == spu_value);                                \
          } else {                                                           \
            equal = (std::abs<TYPE>(xla_value - spu_value) < 1e-2);          \
          }                                                                  \
          if (!equal) {                                                      \
            SPDLOG_INFO(                                                     \
                "Equal check failed at ({}), xla_value = {}, spu_value= {}", \
                fmt::join(output_index, ","), xla_value, spu_value);         \
            pass = false;                                                    \
            ++mismatch;                                                      \
          }                                                                  \
        });                                                                  \
    break;                                                                   \
  }

  switch (expected.eltype().as<PtTy>()->pt_type()) {
    ALL_POSSIBLE_PTTYPES(CASE)

    default:
      YACL_THROW("unexpected type={}", expected.eltype());
  }

  SPDLOG_INFO("Answer has {} elements, {} mismatch found", numel, mismatch);
  return pass;
#undef CASE
}

bool verifyEqual(HalContext *ctx, const xla::Literal &xla_ret,
                 const spu::Value &expected) {
  auto arr = kernel::hal::dump_public(ctx, expected);
  return verifyEqual(xla_ret, arr);
}

}  // namespace

#define SIMPLE_UNARY_VERIFY_IMPL(OpName, XlaOpCode)                       \
  void XlaVerifier::verify(OpName, absl::Span<const spu::Value> operand,  \
                           absl::Span<const spu::Value> expected) {       \
    spu::Value p_operand = operand[0].isPublic()                          \
                               ? operand[0]                               \
                               : kernel::hal::reveal(ctx_, operand[0]);   \
    spu::Value p_expected = expected[0].isPublic()                        \
                                ? expected[0]                             \
                                : kernel::hal::reveal(ctx_, expected[0]); \
    xla::HloEvaluator eval;                                               \
    auto ret = eval.EvaluateElementwiseUnaryOp(                           \
                       XlaOpCode, convertToXlaLiteral(ctx_, p_operand))   \
                   .value();                                              \
    mismatch_handler_(verifyEqual(ctx_, ret, p_expected));                \
  }

SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::AbsOp, xla::HloOpcode::kAbs)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::NegOp, xla::HloOpcode::kNegate)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::LogOp, xla::HloOpcode::kLog)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::Log1pOp, xla::HloOpcode::kLog1p)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::FloorOp, xla::HloOpcode::kFloor)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::CeilOp, xla::HloOpcode::kCeil)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::LogisticOp, xla::HloOpcode::kLogistic)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::TanhOp, xla::HloOpcode::kTanh)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::NotOp, xla::HloOpcode::kNot)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::ExpOp, xla::HloOpcode::kExp)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::Expm1Op, xla::HloOpcode::kExpm1)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::SqrtOp, xla::HloOpcode::kSqrt)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::RsqrtOp, xla::HloOpcode::kRsqrt)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::SignOp, xla::HloOpcode::kSign)
SIMPLE_UNARY_VERIFY_IMPL(mlir::pphlo::RoundOp, xla::HloOpcode::kRoundNearestAfz)

#undef SIMPLE_UNARY_VERIFY_IMPL

void XlaVerifier::verify(mlir::pphlo::ReciprocalOp,
                         absl::Span<const spu::Value> operand,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operand[0].isPublic()
                             ? operand[0]
                             : kernel::hal::reveal(ctx_, operand[0]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  xla::HloEvaluator eval;
  auto ret = eval.EvaluateElementwiseBinaryOp(
                     xla::HloOpcode::kDivide, xlaOnes(ctx_, p_operand),
                     convertToXlaLiteral(ctx_, p_operand))
                 .value();
  mismatch_handler_(verifyEqual(ctx_, ret, p_expected));
}

#define SIMPLE_BINARY_VERIFY_IMPL(OpName, XlaOpCode)                      \
  void XlaVerifier::verify(OpName, absl::Span<const spu::Value> operands, \
                           absl::Span<const spu::Value> expected) {       \
    spu::Value p_lhs = operands[0].isPublic()                             \
                           ? operands[0]                                  \
                           : kernel::hal::reveal(ctx_, operands[0]);      \
    spu::Value p_rhs = operands[1].isPublic()                             \
                           ? operands[1]                                  \
                           : kernel::hal::reveal(ctx_, operands[1]);      \
    spu::Value p_expected = expected[0].isPublic()                        \
                                ? expected[0]                             \
                                : kernel::hal::reveal(ctx_, expected[0]); \
    xla::HloEvaluator eval;                                               \
    auto ret = eval.EvaluateElementwiseBinaryOp(                          \
                       XlaOpCode, convertToXlaLiteral(ctx_, p_lhs),       \
                       convertToXlaLiteral(ctx_, p_rhs))                  \
                   .value();                                              \
    mismatch_handler_(verifyEqual(ctx_, ret, p_expected));                \
  }

SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::AddOp, xla::HloOpcode::kAdd)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::SubtractOp, xla::HloOpcode::kSubtract)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::MulOp, xla::HloOpcode::kMultiply)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::PowOp, xla::HloOpcode::kPower)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::MaxOp, xla::HloOpcode::kMaximum)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::MinOp, xla::HloOpcode::kMinimum)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::AndOp, xla::HloOpcode::kAnd)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::OrOp, xla::HloOpcode::kOr)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::XorOp, xla::HloOpcode::kXor)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::DivOp, xla::HloOpcode::kDivide)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::RemOp, xla::HloOpcode::kRemainder)

SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::ShiftLeftOp, xla::HloOpcode::kShiftLeft)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::ShiftRightArithmeticOp,
                          xla::HloOpcode::kShiftRightArithmetic)
SIMPLE_BINARY_VERIFY_IMPL(mlir::pphlo::ShiftRightLogicalOp,
                          xla::HloOpcode::kShiftRightLogical)

#undef SIMPLE_BINARY_VERIFY_IMPL

#define COMPARISON_VERIFY_IMPL(OpName, CompDir)                           \
  void XlaVerifier::verify(OpName, absl::Span<const spu::Value> operands, \
                           absl::Span<const spu::Value> expected) {       \
    spu::Value p_lhs = operands[0].isPublic()                             \
                           ? operands[0]                                  \
                           : kernel::hal::reveal(ctx_, operands[0]);      \
    spu::Value p_rhs = operands[1].isPublic()                             \
                           ? operands[1]                                  \
                           : kernel::hal::reveal(ctx_, operands[1]);      \
    spu::Value p_expected = expected[0].isPublic()                        \
                                ? expected[0]                             \
                                : kernel::hal::reveal(ctx_, expected[0]); \
    xla::HloEvaluator eval;                                               \
    auto ret = eval.EvaluateElementwiseCompareOp(                         \
                       CompDir, convertToXlaLiteral(ctx_, p_lhs),         \
                       convertToXlaLiteral(ctx_, p_rhs))                  \
                   .value();                                              \
    mismatch_handler_(verifyEqual(ctx_, ret, p_expected));                \
  }

COMPARISON_VERIFY_IMPL(mlir::pphlo::EqualOp, xla::ComparisonDirection::kEq)
COMPARISON_VERIFY_IMPL(mlir::pphlo::NotEqualOp, xla::ComparisonDirection::kNe)
COMPARISON_VERIFY_IMPL(mlir::pphlo::LessOp, xla::ComparisonDirection::kLt)
COMPARISON_VERIFY_IMPL(mlir::pphlo::LessEqualOp, xla::ComparisonDirection::kLe)
COMPARISON_VERIFY_IMPL(mlir::pphlo::GreaterOp, xla::ComparisonDirection::kGt)
COMPARISON_VERIFY_IMPL(mlir::pphlo::GreaterEqualOp,
                       xla::ComparisonDirection::kGe)

#undef COMPARISON_VERIFY_IMPL

void XlaVerifier::verify(mlir::pphlo::DotOp,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_lhs = operands[0].isPublic()
                         ? operands[0]
                         : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_rhs = operands[1].isPublic()
                         ? operands[1]
                         : kernel::hal::reveal(ctx_, operands[1]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  xla::HloEvaluator eval;
  xla::DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(p_lhs.shape().size() == 1 ? 0 : 1);
  dnums.add_rhs_contracting_dimensions(0);
  auto ret = eval.EvaluateDotOp(dnums, xla::PrecisionConfig::default_instance(),
                                convertToXlaLiteral(ctx_, p_lhs),
                                convertToXlaLiteral(ctx_, p_rhs))
                 .value();
  mismatch_handler_(verifyEqual(ctx_, ret, p_expected));
}

void XlaVerifier::verify(mlir::pphlo::DotGeneralOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_lhs = operands[0].isPublic()
                         ? operands[0]
                         : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_rhs = operands[1].isPublic()
                         ? operands[1]
                         : kernel::hal::reveal(ctx_, operands[1]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  xla::HloEvaluator eval;
  xla::DotDimensionNumbers dnums;

  for (auto d : op.getDotDimensionNumbers().getLhsBatchingDimensions()) {
    dnums.add_lhs_batch_dimensions(d);
  }

  for (auto d : op.getDotDimensionNumbers().getLhsContractingDimensions()) {
    dnums.add_lhs_contracting_dimensions(d);
  }

  for (auto d : op.getDotDimensionNumbers().getRhsBatchingDimensions()) {
    dnums.add_rhs_batch_dimensions(d);
  }

  for (auto d : op.getDotDimensionNumbers().getRhsContractingDimensions()) {
    dnums.add_rhs_contracting_dimensions(d);
  }

  auto ret = eval.EvaluateDotOp(dnums, xla::PrecisionConfig::default_instance(),
                                convertToXlaLiteral(ctx_, p_lhs),
                                convertToXlaLiteral(ctx_, p_rhs))
                 .value();

  mismatch_handler_(verifyEqual(ctx_, ret, p_expected));
}

void XlaVerifier::verify(mlir::pphlo::SelectOp,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_pred = operands[0].isPublic()
                          ? operands[0]
                          : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_on_true = operands[1].isPublic()
                             ? operands[1]
                             : kernel::hal::reveal(ctx_, operands[1]);
  spu::Value p_on_false = operands[2].isPublic()
                              ? operands[2]
                              : kernel::hal::reveal(ctx_, operands[2]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  xla::HloEvaluator eval;
  auto ret = eval.EvaluateElementwiseTernaryOp(
                     xla::HloOpcode::kSelect, convertToXlaLiteral(ctx_, p_pred),
                     convertToXlaLiteral(ctx_, p_on_true),
                     convertToXlaLiteral(ctx_, p_on_false))
                 .value();
  mismatch_handler_(verifyEqual(ctx_, ret, p_expected));
}

void XlaVerifier::verify(mlir::pphlo::ClampOp,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_min = operands[0].isPublic()
                         ? operands[0]
                         : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_operand = operands[1].isPublic()
                             ? operands[1]
                             : kernel::hal::reveal(ctx_, operands[1]);
  spu::Value p_max = operands[2].isPublic()
                         ? operands[2]
                         : kernel::hal::reveal(ctx_, operands[2]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  xla::HloEvaluator eval;
  auto ret = eval.EvaluateElementwiseTernaryOp(
                     xla::HloOpcode::kClamp, convertToXlaLiteral(ctx_, p_min),
                     convertToXlaLiteral(ctx_, p_operand),
                     convertToXlaLiteral(ctx_, p_max))
                 .value();
  mismatch_handler_(verifyEqual(ctx_, ret, p_expected));
}

void XlaVerifier::verify(mlir::pphlo::ConvolutionOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_lhs = operands[0].isPublic()
                         ? operands[0]
                         : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_rhs = operands[1].isPublic()
                         ? operands[1]
                         : kernel::hal::reveal(ctx_, operands[1]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  auto lhs_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_lhs));
  auto rhs_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_rhs));

  xla::Window window;
  for (size_t idx = 0;
       idx < op.getDimensionNumbers().getKernelSpatialDimensions().size();
       ++idx) {
    auto *w = window.add_dimensions();
    w->set_size(p_rhs.shape()[op.getDimensionNumbers()
                                  .getKernelSpatialDimensions()[idx]]);
    w->set_stride(op.getWindowStrides().has_value()
                      ? op.getWindowStrides()->getValues<int64_t>()[idx]
                      : 1);
    w->set_base_dilation(1);
    w->set_window_dilation(1);
    w->set_padding_low(0);
    w->set_padding_high(0);
    w->set_window_reversal(false);
  }

  xla::ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(
      op.getDimensionNumbers().getInputBatchDimension());
  dnums.set_input_feature_dimension(
      op.getDimensionNumbers().getInputFeatureDimension());
  for (auto v : op.getDimensionNumbers().getInputSpatialDimensions()) {
    dnums.add_input_spatial_dimensions(v);
  }

  dnums.set_kernel_input_feature_dimension(
      op.getDimensionNumbers().getKernelInputFeatureDimension());
  dnums.set_kernel_output_feature_dimension(
      op.getDimensionNumbers().getKernelOutputFeatureDimension());

  for (auto v : op.getDimensionNumbers().getKernelSpatialDimensions()) {
    dnums.add_kernel_spatial_dimensions(v);
  }

  dnums.set_output_batch_dimension(
      op.getDimensionNumbers().getOutputBatchDimension());
  dnums.set_output_feature_dimension(
      op.getDimensionNumbers().getOutputFeatureDimension());

  for (auto v : op.getDimensionNumbers().getOutputSpatialDimensions()) {
    dnums.add_output_spatial_dimensions(v);
  }

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);
  auto cloned_instruction = xla::HloInstruction::CreateConvolve(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      lhs_instr.get(), rhs_instr.get(), op.getFeatureGroupCount(),
      op.getBatchGroupCount(), window, dnums,
      xla::PrecisionConfig::default_instance());

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());

  mismatch_handler_(verifyEqual(result.value(), expected_arr));
}

void XlaVerifier::verify(mlir::pphlo::DynamicSliceOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  std::vector<spu::Value> p_start_indicies(operands.size() - 1);

  for (size_t idx = 1; idx < operands.size(); ++idx) {
    p_start_indicies[idx - 1] = operands[idx].isPublic()
                                    ? operands[idx]
                                    : kernel::hal::reveal(ctx_, operands[idx]);
  }

  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);

  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));
  std::vector<std::unique_ptr<xla::HloInstruction>> start_indicies_instrs;
  std::vector<xla::HloInstruction *> start_indicies_instrs_ptr;
  for (const auto &s : p_start_indicies) {
    start_indicies_instrs.emplace_back(
        xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, s)));
    start_indicies_instrs_ptr.emplace_back(start_indicies_instrs.back().get());
  }

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);
  auto cloned_instruction = xla::HloInstruction::CreateDynamicSlice(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get(), start_indicies_instrs_ptr,
      std::vector<int64_t>(op.getSliceSizes().getValues<int64_t>().begin(),
                           op.getSliceSizes().getValues<int64_t>().end()));

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());

  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::DynamicUpdateSliceOp,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_update = operands[1].isPublic()
                            ? operands[1]
                            : kernel::hal::reveal(ctx_, operands[1]);

  std::vector<spu::Value> p_start_indicies(operands.size() - 2);
  for (size_t idx = 2; idx < operands.size(); ++idx) {
    p_start_indicies[idx - 2] = operands[idx].isPublic()
                                    ? operands[idx]
                                    : kernel::hal::reveal(ctx_, operands[idx]);
  }

  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);

  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));
  auto update_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_update));
  std::vector<std::unique_ptr<xla::HloInstruction>> start_indicies_instrs;
  std::vector<xla::HloInstruction *> start_indicies_instrs_ptr;
  for (const auto &s : p_start_indicies) {
    start_indicies_instrs.emplace_back(
        xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, s)));
    start_indicies_instrs_ptr.emplace_back(start_indicies_instrs.back().get());
  }

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);
  auto cloned_instruction = xla::HloInstruction::CreateDynamicUpdateSlice(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get(), update_instr.get(), start_indicies_instrs_ptr);

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());

  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::GatherOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_start_indicies = operands[1].isPublic()
                                    ? operands[1]
                                    : kernel::hal::reveal(ctx_, operands[1]);

  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);

  xla::GatherDimensionNumbers dnums;
  for (const auto &o : op.getDimensionNumbers().getOffsetDims()) {
    dnums.add_offset_dims(o);
  }
  for (const auto &o : op.getDimensionNumbers().getCollapsedSliceDims()) {
    dnums.add_collapsed_slice_dims(o);
  }
  for (const auto &o : op.getDimensionNumbers().getStartIndexMap()) {
    dnums.add_start_index_map(o);
  }
  dnums.set_index_vector_dim(op.getDimensionNumbers().getIndexVectorDim());

  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));
  auto start_indicies_instr = xla::HloInstruction::CreateConstant(
      convertToXlaLiteral(ctx_, p_start_indicies));

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);
  auto cloned_instruction = xla::HloInstruction::CreateGather(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get(), start_indicies_instr.get(), dnums,
      std::vector<int64_t>(op.getSliceSizes().getValues<int64_t>().begin(),
                           op.getSliceSizes().getValues<int64_t>().end()),
      op.getIndicesAreSorted());

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());

  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::ReshapeOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);
  auto cloned_instruction = xla::HloInstruction::CreateReshape(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get());

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());
  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::BroadcastOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);

  auto cloned_instruction = xla::HloInstruction::CreateBroadcast(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get(),
      convertDenseIntElementAttr(op.getBroadcastDimensions()));

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());
  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::TransposeOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);

  auto cloned_instruction = xla::HloInstruction::CreateTranspose(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get(), convertDenseIntElementAttr(op.getPermutation()));

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());
  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::IotaOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);

  auto cloned_instruction = xla::HloInstruction::CreateIota(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      op.getIotaDimension());

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());
  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::SliceOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);
  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);

  auto cloned_instruction = xla::HloInstruction::CreateSlice(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get(), convertDenseIntElementAttr(op.getStartIndices()),
      convertDenseIntElementAttr(op.getLimitIndices()),
      convertDenseIntElementAttr(op.getStrides()));

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());
  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::ConcatenateOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);

  std::vector<std::unique_ptr<xla::HloInstruction>> operand_instrs(
      operands.size());
  std::vector<xla::HloInstruction *> ops(operand_instrs.size());
  for (size_t idx = 0; idx < operands.size(); ++idx) {
    spu::Value p_operand = operands[idx].isPublic()
                               ? operands[idx]
                               : kernel::hal::reveal(ctx_, operands[idx]);
    operand_instrs[idx] = xla::HloInstruction::CreateConstant(
        convertToXlaLiteral(ctx_, p_operand));
    ops[idx] = operand_instrs[idx].get();
  }

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);

  auto cloned_instruction = xla::HloInstruction::CreateConcatenate(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()), ops,
      op.getDimension());

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());
  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::PadOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_pad_value = operands[1].isPublic()
                               ? operands[1]
                               : kernel::hal::reveal(ctx_, operands[1]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);

  auto operand_instr =
      xla::HloInstruction::CreateConstant(convertToXlaLiteral(ctx_, p_operand));
  auto pad_value_instr = xla::HloInstruction::CreateConstant(
      convertToXlaLiteral(ctx_, p_pad_value));

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);

  xla::PaddingConfig config;
  for (int64_t i = 0; i < op.getEdgePaddingLow().size(); ++i) {
    auto *dims = config.add_dimensions();
    dims->set_edge_padding_low(op.getEdgePaddingLow().getValues<int64_t>()[i]);
    dims->set_edge_padding_high(
        op.getEdgePaddingHigh().getValues<int64_t>()[i]);
    dims->set_interior_padding(op.getInteriorPadding().getValues<int64_t>()[i]);
  }

  auto cloned_instruction = xla::HloInstruction::CreatePad(
      buildXLAShape(*expected_arr.eltype().as<PtTy>(), p_expected.shape()),
      operand_instr.get(), pad_value_instr.get(), config);

  xla::HloEvaluator eval;
  auto result = eval.Evaluate(cloned_instruction.get());
  mismatch_handler_(verifyEqual(ctx_, result.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::BitcastConvertOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  // Nothing to verify
  SPDLOG_INFO("No verification method implemented");
}

void XlaVerifier::verify(mlir::pphlo::ConvertOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  mlir::pphlo::TypeTools typetools;
  auto in_vis = typetools.getTypeVisibility(op.getOperand().getType());
  auto ret_vis = typetools.getTypeVisibility(op->getResultTypes()[0]);

  if (in_vis != ret_vis) {
    SPDLOG_INFO("Visibility cast, nothing to verify");
    return;
  }

  spu::Value p_operand = operands[0].isPublic()
                             ? operands[0]
                             : kernel::hal::reveal(ctx_, operands[0]);
  spu::Value p_expected = expected[0].isPublic()
                              ? expected[0]
                              : kernel::hal::reveal(ctx_, expected[0]);

  auto operand_xla = convertToXlaLiteral(ctx_, p_operand);

  const auto expected_arr = kernel::hal::dump_public(ctx_, p_expected);

  auto converted =
      operand_xla.Convert(getXlaType(*expected_arr.eltype().as<PtTy>()));
  mismatch_handler_(verifyEqual(ctx_, converted.value(), p_expected));
}

void XlaVerifier::verify(mlir::pphlo::ReduceOp op,
                         absl::Span<const spu::Value> operands,
                         absl::Span<const spu::Value> expected) {
  // Nothing to verify
  SPDLOG_INFO("No verification method implemented");
}

#define UNIMPL_VERIFIER(OpName)          \
  void XlaVerifier::verify(OpName op, \ 
                         absl::Span<const spu::Value> operands, \ 
                         absl::Span<const spu::Value> expected) { \
    YACL_THROW("TBD");                   \
  }

UNIMPL_VERIFIER(mlir::pphlo::SelectAndScatterOp)
UNIMPL_VERIFIER(mlir::pphlo::ReverseOp)

UNIMPL_VERIFIER(mlir::pphlo::ReduceWindowOp)

UNIMPL_VERIFIER(mlir::pphlo::SortOp)

}  // namespace spu::device::pphlo
