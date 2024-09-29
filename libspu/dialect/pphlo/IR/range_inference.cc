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

#include <cstdint>

#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/utils/utils.h"

namespace mlir::spu::pphlo {

namespace {

ConstantIntRanges changeWidth(const ConstantIntRanges& in, size_t out_width) {
  if (in.smax().getBitWidth() == out_width) {
    return in;
  }

  // Bool is defacto an unsigned thingy
  if (in.smax().getBitWidth() == 1) {
    return ConstantIntRanges(APInt(out_width, in.umin().getZExtValue()),
                             APInt(out_width, in.umax().getZExtValue()),
                             APInt(out_width, in.smin().getZExtValue()),
                             APInt(out_width, in.smax().getZExtValue()));
  }

  if (out_width == 128) {
    // Need manual extend width
    int128_t umin = in.umin().getZExtValue();
    int128_t umax = in.umax().getZExtValue();
    int128_t smin = in.smin().getSExtValue();
    int128_t smax = in.smax().getSExtValue();

    return ConstantIntRanges(
        convertFromInt128(128, umin), convertFromInt128(128, umax),
        convertFromInt128(128, smin), convertFromInt128(128, smax));
  }

  if (in.umax().getBitWidth() == 128) {
    // From 128 to something else
    int128_t umin = convertFromAPInt(in.umin());
    int128_t umax = convertFromAPInt(in.umax());
    int128_t smin = convertFromAPInt(in.smin());
    int128_t smax = convertFromAPInt(in.smax());

    return ConstantIntRanges(APInt(out_width, umin), APInt(out_width, umax),
                             APInt(out_width, smin), APInt(out_width, smax));
  }

  return ConstantIntRanges(APInt(out_width, in.umin().getZExtValue()),
                           APInt(out_width, in.umax().getZExtValue()),
                           APInt(out_width, in.smin().getSExtValue()),
                           APInt(out_width, in.smax().getSExtValue()));
}

inline APInt GetSminValue(ArrayRef<ConstantIntRanges> args_range,
                          size_t begin = 0, size_t end = SIZE_MAX) {
  end = std::min(end, args_range.size());
  SPU_ENFORCE(begin < end);

  auto min = args_range[begin].smin();
  while (++begin < end) {
    if (const auto& tmp = args_range[begin].smin(); !min.sle(tmp)) {
      min = tmp;
    }
  }

  return min;
}

inline APInt GetSmaxValue(ArrayRef<ConstantIntRanges> args_range,
                          size_t begin = 0, size_t end = SIZE_MAX) {
  end = std::min(end, args_range.size());
  SPU_ENFORCE(begin < end);

  auto max = args_range[begin].smax();
  while (++begin < end) {
    if (const auto& tmp = args_range[begin].smax(); !max.sle(tmp)) {
      max = tmp;
    }
  }

  return max;
}

}  // namespace

void SignOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                               SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& in_range = args_range[0];

  int128_t result_max = 1;
  int128_t result_min = -1;
  bool ignore_zero = getIgnoreZero();

  if (in_range.smax().isNonPositive()) {
    if (!ignore_zero && in_range.smax().isZero()) {
      result_max = 0;
    } else {
      result_max = -1;
    }
  }

  if (in_range.smin().isNonNegative()) {
    if (!ignore_zero && in_range.smin().isZero()) {
      result_min = 0;
    } else {
      result_min = 1;
    }
  }

  TypeTools tools(getContext());
  if (tools.isFixedPointType(getType())) {
    int64_t scale = 1 << tools.getFxpBits(getType());
    result_max *= scale;
    result_min *= scale;
  }

  auto width = tools.getIntOrFxpWidth(getType());

  APInt current_max = convertFromInt128(width, result_max);
  APInt current_min = convertFromInt128(width, result_min);

  setResultRange(getResult(),
                 ConstantIntRanges::fromSigned(current_min, current_max));
}

void MulOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                              SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  auto lhs_range = args_range[0];
  auto rhs_range = args_range[1];

  auto makeAbsRange = [](const ConstantIntRanges& range) {
    auto low_abs = range.smin().abs();
    auto high_abs = range.smax().abs();

    return ConstantIntRanges::fromSigned(
        APInt::getZero(low_abs.getBitWidth()),
        high_abs.sgt(low_abs) ? high_abs : low_abs);
  };

  // A special case for sign*v
  if (auto sign = getLhs().getDefiningOp<SignOp>()) {
    if (sign.getOperand() == getRhs()) {
      // range is |rhs_range|
      setResultRange(getResult(), makeAbsRange(rhs_range));
      return;
    }
  }

  if (auto sign = getRhs().getDefiningOp<SignOp>()) {
    if (sign.getOperand() == getLhs()) {
      // range is |rhs_range|
      setResultRange(getResult(), makeAbsRange(lhs_range));
      return;
    }
  }

  auto out_bitwidth =
      std::max(lhs_range.smin().getBitWidth(), rhs_range.smin().getBitWidth());

  lhs_range = changeWidth(lhs_range, out_bitwidth);
  rhs_range = changeWidth(rhs_range, out_bitwidth);

  APInt smax, smin, umax, umin;
  // Process signed range
  {
    smax = APInt::getSignedMinValue(out_bitwidth);
    smin = APInt::getSignedMaxValue(out_bitwidth);

    llvm::SmallVector<APInt, 4> values{
        lhs_range.smax().smul_sat(rhs_range.smax()),
        lhs_range.smax().smul_sat(rhs_range.smin()),
        lhs_range.smin().smul_sat(rhs_range.smax()),
        lhs_range.smin().smul_sat(rhs_range.smin())};

    for (const auto& v : values) {
      if (v.sgt(smax)) {
        smax = v;
      }
      if (v.slt(smin)) {
        smin = v;
      }
    }
  }
  // Process unsigned range
  {
    umax = APInt::getMinValue(out_bitwidth);
    umin = APInt::getMaxValue(out_bitwidth);

    llvm::SmallVector<APInt, 4> values{
        lhs_range.umax().umul_sat(rhs_range.umax()),
        lhs_range.umax().umul_sat(rhs_range.umin()),
        lhs_range.umin().umul_sat(rhs_range.umax()),
        lhs_range.umin().umul_sat(rhs_range.umin())};

    for (const auto& v : values) {
      if (v.ugt(umax)) {
        umax = v;
      }
      if (v.ult(umin)) {
        umin = v;
      }
    }
  }

  setResultRange(getResult(), ConstantIntRanges(umin, umax, smin, smax));
}

void ConvertOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                  SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& in_range = args_range[0];

  auto source_type = getOperand().getType().getElementType();
  auto dest_type = getResult().getType().getElementType();

  TypeTools tools(getContext());

  // cast from fxp source to dest
  if (tools.isIntType(dest_type) && tools.isFixedPointType(source_type)) {
    auto width = tools.getIntOrFxpWidth(dest_type);
    auto fxp_bits = tools.getFxpBits(source_type);
    APInt scale(in_range.smax().getBitWidth(), 1 << fxp_bits);

    auto umin = in_range.umin().udiv(scale);
    auto umax = in_range.umax().udiv(scale);
    auto smin = in_range.smin().sdiv(scale);
    auto smax = in_range.smax().sdiv(scale);

    ConstantIntRanges out_range(umin, umax, smin, smax);

    setResultRange(getResult(), changeWidth(out_range, width));

    return;
  }

  // cast from int to fxp dest
  if (tools.isIntType(source_type) && tools.isFixedPointType(dest_type)) {
    auto width = tools.getIntOrFxpWidth(dest_type);
    auto fxp_bits = tools.getFxpBits(dest_type);
    APInt scale(width, 1 << fxp_bits);
    auto is_unsigned =
        dyn_cast<IntegerType>(tools.getBaseType(source_type)).isUnsigned();

    // Switch bits first
    auto extend_in = changeWidth(in_range, width);

    auto umin = extend_in.umin().umul_sat(scale);
    auto umax = extend_in.umax().umul_sat(scale);
    auto smin = extend_in.smin().smul_sat(scale);
    auto smax = extend_in.smax().smul_sat(scale);

    if (is_unsigned) {
      smin = APInt::getZero(width);
    }

    setResultRange(getResult(), ConstantIntRanges(umin, umax, smin, smax));

    return;
  }

  auto out_width = tools.getIntOrFxpWidth(getType());
  if (out_width == in_range.smax().getBitWidth()) {
    setResultRange(getResult(), args_range[0]);
    return;
  }

  // Expand bits
  setResultRange(getResult(), changeWidth(in_range, out_width));
}

void BitcastConvertOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                         SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  // bitcast conver cannot change width of type, so no range diff
  setResultRange(getResult(), args_range[0]);
}

void PrefixOrOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                   SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& in_range = args_range[0];

  // Is full range
  if (in_range.umax().isMaxValue() && in_range.umin().isZero()) {
    setResultRange(getResult(), in_range);
    return;
  }

  auto computePrefixOrRange = [](const APInt& in) {
    auto bits = in.getActiveBits();
    return APInt(in.getBitWidth(), std::pow(2, bits) - 1);
  };

  auto high = in_range.umax();
  if (!high.isMaxValue()) {
    high = computePrefixOrRange(high);
  }

  auto low = in_range.umin();

  if (!low.isZero()) {
    low = computePrefixOrRange(low);
  }

  return setResultRange(getResult(),
                        ConstantIntRanges::fromUnsigned(low, high));
}

void ShiftRightLogicalOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> args_range, SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  const auto& in_range = args_range[0];
  const auto& shift_range = args_range[1];

  // normal shift
  if (shift_range.getConstantValue().has_value()) {
    auto shift_value = *shift_range.getConstantValue();

    setResultRange(getResult(),
                   ConstantIntRanges(in_range.umin().lshr(shift_value),
                                     in_range.umax().lshr(shift_value),
                                     in_range.smin().lshr(shift_value),
                                     in_range.smax().lshr(shift_value)));

    return;
  }

  // strange crazy shift...be conservative
  setResultRange(getResult(),
                 ConstantIntRanges(in_range.umin().lshr(shift_range.umax()),
                                   in_range.umax().lshr(shift_range.umin()),
                                   in_range.smin().lshr(shift_range.umax()),
                                   in_range.smax().lshr(shift_range.umin())));
}

void XorOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                              SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  const auto& lhs_range = args_range[0];
  const auto& rhs_range = args_range[1];

  auto bitwidth = lhs_range.umax().getBitWidth();

  auto active_bits = std::max(lhs_range.umax().getActiveBits(),
                              rhs_range.umax().getActiveBits());

  auto low = APInt::getZero(bitwidth);
  auto high = APInt::getAllOnes(bitwidth).lshr(bitwidth - active_bits);

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(low, high));
}

void AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                              SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  const auto& lhs_range = args_range[0];
  const auto& rhs_range = args_range[1];

  auto bitwidth = lhs_range.umax().getBitWidth();

  auto active_bits = std::min(lhs_range.umax().getActiveBits(),
                              rhs_range.umax().getActiveBits());

  auto low = APInt::getZero(bitwidth);
  auto high = APInt::getAllOnes(bitwidth).lshr(bitwidth - active_bits);

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(low, high));
}

void OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                             SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  const auto& lhs_range = args_range[0];
  const auto& rhs_range = args_range[1];

  auto bitwidth = lhs_range.umax().getBitWidth();

  auto active_bits = std::max(lhs_range.umax().getActiveBits(),
                              rhs_range.umax().getActiveBits());

  auto low = APInt::getZero(bitwidth);
  auto high = APInt::getAllOnes(bitwidth).lshr(bitwidth - active_bits);

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(low, high));
}

void BitRevOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                 SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& in_range = args_range[0];

  if (in_range.umax().getActiveBits() > getEnd()) {
    setResultRange(getResult(), in_range);
    return;
  }

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(
                                  APInt::getZero(in_range.umax().getBitWidth()),
                                  APInt(in_range.umax().getBitWidth(),
                                        std::pow(2, getEnd()) - 1)));
}

void TruncOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& in_range = args_range[0];

  TypeTools tools(getContext());

  auto in_fxp_bits = tools.getFxpBits(getOperand().getType());
  auto out_fxp_bits = tools.getFxpBits(getType());

  auto trunc_bits = in_fxp_bits - out_fxp_bits;

  setResultRange(getResult(), ConstantIntRanges::fromSigned(
                                  in_range.smin().ashr(trunc_bits),
                                  in_range.smax().ashr(trunc_bits)));
}

void NegOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                              SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& in_range = args_range[0];

  auto high = in_range.smin().isMinSignedValue()
                  ? APInt::getSignedMaxValue(in_range.smin().getBitWidth())
                  : -in_range.smin();
  auto low = -in_range.smax();

  setResultRange(getResult(), ConstantIntRanges::fromSigned(low, high));
}

void AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                              SetIntRangeFn setResultRange) {
  auto arg_cnt = args_range.size();
  SPU_ENFORCE(arg_cnt >= 1);

  if (1 == arg_cnt) {
    setResultRange(getResult(), args_range[0]);
    return;
  }

  const auto& lhs_range = args_range[0];
  const auto& rhs_range = args_range[1];

  setResultRange(getResult(), ConstantIntRanges::fromSigned(
                                  lhs_range.smin().sadd_sat(rhs_range.smin()),
                                  lhs_range.smax().sadd_sat(rhs_range.smax())));
}

void LessOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 ConstantIntRanges::fromSigned(APInt::getZero(1), APInt(1, 1)));
}

void NotOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                              SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& in_range = args_range[0];

  TypeTools tools(getContext());

  auto width = tools.getIntWidth(getType());
  if (width == 1) {
    setResultRange(getResult(), in_range);
  } else {
    auto smax = APInt::getSignedMaxValue(width);
    auto smin = APInt::getSignedMinValue(width);
    setResultRange(getResult(), ConstantIntRanges::fromSigned(smin, smax));
  }
}

void PadOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                              SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  // 0-th: input data
  // 1-th: padding value
  auto min = GetSminValue(args_range, 0, 2);
  auto max = GetSmaxValue(args_range, 0, 2);

  setResultRange(getResult(), ConstantIntRanges::fromSigned(min, max));
}

void BitDeintlOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                    SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  // split the bits via odd and even, and then concate them, such as
  // 0b1010,1010 -> 0b1111,0000
  auto width = TypeTools(getContext()).getIntWidth(getType());
  auto min = APInt::getMinValue(width);
  auto max = APInt::getMaxValue(width);

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}

void BroadcastOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                    SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);
  setResultRange(getResult(), args_range[0]);
}

void ConcatenateOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                      SetIntRangeFn setResultRange) {
  SPU_ENFORCE(!args_range.empty());

  auto min = GetSminValue(args_range);
  auto max = GetSmaxValue(args_range);

  setResultRange(getResult(), ConstantIntRanges::fromSigned(min, max));
}

void EqualOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 ConstantIntRanges::fromSigned(APInt::getZero(1), APInt(1, 1)));
}

void ReshapeOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                  SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);
  setResultRange(getResult(), args_range[0]);
}

void ReverseOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                  SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);
  setResultRange(getResult(), args_range[0]);
}

void ShiftLeftOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                    SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  const auto& in_range = args_range[0];
  const auto& shift_range = args_range[1];

  // normal shift
  if (shift_range.getConstantValue().has_value()) {
    auto shift_value = *shift_range.getConstantValue();

    setResultRange(getResult(),
                   ConstantIntRanges(in_range.umin().shl(shift_value),
                                     in_range.umax().shl(shift_value),
                                     in_range.smin().shl(shift_value),
                                     in_range.smax().shl(shift_value)));

    return;
  }

  // strange crazy shift...be conservative
  setResultRange(getResult(),
                 ConstantIntRanges(in_range.umin().shl(shift_range.umin()),
                                   in_range.umax().shl(shift_range.umax()),
                                   in_range.smin().shl(shift_range.umax()),
                                   in_range.smax().shl(shift_range.umin())));
}

void ShiftRightArithmeticOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> args_range, SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 2);

  const auto& in_range = args_range[0];
  const auto& shift_range = args_range[1];

  // normal shift
  if (shift_range.getConstantValue().has_value()) {
    auto shift_value = *shift_range.getConstantValue();

    setResultRange(getResult(),
                   ConstantIntRanges(in_range.umin().ashr(shift_value),
                                     in_range.umax().ashr(shift_value),
                                     in_range.smin().ashr(shift_value),
                                     in_range.smax().ashr(shift_value)));

    return;
  }

  const auto& shift_min = shift_range.umin();
  const auto& shfit_max = shift_range.umax();

  auto umin = in_range.umin();
  auto umax = in_range.umax();

  // case0: sign is one, ashr will fill one,
  //        such as, 0x8 (0b1000) --(1)--> 0xc(0b1100)
  // case1: sign is zero, ashr will fill zero,
  //        such as, 0x7 (0b0111) --(1)--> 0x3(0b0110)
  umin = umin.isNegative() ? umin.ashr(shift_min) : umin.ashr(shfit_max);
  umax = umax.isNegative() ? umax.ashr(shfit_max) : umax.ashr(shift_min);

  // strange crazy shift...be conservative
  setResultRange(getResult(),
                 ConstantIntRanges(umin, umax, in_range.smin().ashr(shfit_max),
                                   in_range.smax().ashr(shift_min)));
}

void SliceOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);
  setResultRange(getResult(), args_range[0]);
}

void DynamicSliceOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                       SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);
  setResultRange(getResult(), args_range[0]);
}

void DynamicUpdateSliceOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> args_range, SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);
  setResultRange(getResult(), args_range[0]);
}

void SqrtOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                               SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);

  const auto& umax = args_range[0].umax();
  setResultRange(getResult(),
                 ConstantIntRanges::fromSigned(
                     APInt::getZero(umax.getBitWidth()), umax.sqrt()));
}

void TransposeOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                                    SetIntRangeFn setResultRange) {
  SPU_ENFORCE(args_range.size() >= 1);
  setResultRange(getResult(), args_range[0]);
}

// TODO: how to range inference for IfOp, CaseOp, WhileOp and ReduceOp?
// some viewpoints:
// 1. Because of AbstractSparseForwardDataFlowAnalysis::initializeRecursively
//    will visit the body function of WhileOp/ReduceOp, we don't need a range
//    inference for these ops.
// 2. Since it's currently unused and there isn't a good solution for it, let's
//    ignore it for now.
#if 0
void CaseOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                               SetIntRangeFn setResultRange) {
  auto arg_cnt = args_range.size();
  SPU_ENFORCE(arg_cnt >= 2);

  if (auto pred = args_range[0].getConstantValue(); pred.has_value()) {
    auto branch_idx = pred->getRawData()[0];
    branch_idx = (branch_idx + arg_cnt) % arg_cnt;

    // the inputs of branch
    for (unsigned i = 0; i < getNumResults(); ++i) {
      setResultRange(getResult(i),
                     ConstantIntRanges(args_range[1 + branch_idx]));
    }

    return;
  }

  auto min = GetSminValue(args_range, 1 /*begin*/);
  auto max = GetSmaxValue(args_range, 1 /*begin*/);

  for (unsigned i = 0; i < getNumResults(); ++i) {
    setResultRange(getResult(i), ConstantIntRanges::fromSigned(min, max));
  }
}

void IfOp::inferResultRanges(ArrayRef<ConstantIntRanges> args_range,
                             SetIntRangeFn setResultRange) {
  auto args_cnt = args_range.size();
  SPU_ENFORCE(1 == args_cnt || 3 == args_cnt);

  if (1 == args_cnt) {
    // do nothing, range infer by visiting body function
    return;
  }

  if (auto pred = args_range[0].getConstantValue(); pred.has_value()) {
    auto idx = pred->getBoolValue() ? 1 : 2;

    for (unsigned i = 0; i < getNumResults(); ++i) {
      setResultRange(getResult(i), ConstantIntRanges(args_range[idx]));
    }

    return;
  }

  auto min = GetSminValue(args_range, 1 /*begin*/);
  auto max = GetSmaxValue(args_range, 1 /*begin*/);

  for (unsigned i = 0; i < getNumResults(); ++i) {
    setResultRange(getResult(i), ConstantIntRanges::fromSigned(min, max));
  }
}
#endif  // 0

}  // namespace mlir::spu::pphlo
