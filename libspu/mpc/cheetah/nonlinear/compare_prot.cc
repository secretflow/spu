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

#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"

#include "emp-tool/utils/prg.h"
#include "yacl/link/link.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ferret.h"
#include "libspu/mpc/cheetah/ot/util.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

CompareProtocol::CompareProtocol(std::shared_ptr<BasicOTProtocols> base)
    : basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
  is_sender_ = base->Rank() == 0;
}

CompareProtocol::~CompareProtocol() { basic_ot_prot_->Flush(); }

void SetLeafOTMsg(absl::Span<uint8_t> ot_messages, uint8_t digit,
                  uint8_t rnd_cmp_bit, uint8_t rnd_eq_bit, bool gt) {
  uint8_t N = ot_messages.size();
  SPU_ENFORCE(digit < N);
  for (uint8_t i = 0; i < N; i++) {
    if (gt) {
      ot_messages[i] = rnd_cmp_bit ^ static_cast<uint8_t>(digit > i);
    } else {
      ot_messages[i] = rnd_cmp_bit ^ static_cast<uint8_t>(digit < i);
    }

    // compact two bits into one OT message
    ot_messages[i] |= ((rnd_eq_bit ^ static_cast<uint8_t>(digit == i)) << 1);
  }
}

// x0, x1, ..., x{n-1}
// y0, y1, ..., y{n-1}
// Decomposite into digits
// x0 -> x{0, 0}, ..., x{0, M-1} digits
// x1 -> x{1, 0}, ..., x{1, M-1} digits
// ...
// x{n-1} -> x{n-1, 0}, ..., x{n-1, M-1} digits
// Each x{i, j} expands to x{i, j, 0}, x{i, j, 1}..., x{i, j, N-1} booleans
//   such that x{i, j, s} = 1(x{i, j} < s) and M = 2^N
//
// y0 -> y{1, 0}, ..., y{1, M-1} digits
// ...
// y{n-1} -> y{n-1, 0}, ..., y{n-1, M-1} digits
//
// 1-of-N OT
// Each y{i, j} selects on x{i, j, 0}, x{i, j, 1}, ..., x{i, j, N-1}
// After batched 1-of-N OT
// We obtain
// lt_{i, j} = 1(x{i, j} < y{i, j}) and eq_{i, j} = 1{x{i, j} = y{i, j})
// for each (i, j) pair.
//
ArrayRef CompareProtocol::DoCompute(const ArrayRef& inp, bool greater_than,
                                    ArrayRef* keep_eq) {
  auto field = inp.eltype().as<Ring2k>()->field();
  size_t bit_width = SizeOf(field) * 8;

  size_t remain = bit_width % kCompareRadix;
  size_t num_digits = CeilDiv(bit_width, kCompareRadix);
  size_t radix = 1U << kCompareRadix;  // one-of-N OT

  size_t numel = inp.numel();
  size_t num_cmp = CeilDiv<size_t>(numel, 8U) * 8;
  // init to all zero
  std::vector<uint8_t> digits(num_cmp * num_digits, 0);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(kCompareRadix);
    const auto mask_remain = makeBitsMask<u2k>(remain);
    ArrayView<u2k> xinp(inp);

    for (size_t i = 0; i < numel; ++i) {
      for (size_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * kCompareRadix;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
        // last digits
        if (remain > 0 && (j + 1 == num_digits)) {
          digits[i * num_digits + j] &= mask_remain;
        }
      }
    }
  });

  std::vector<uint8_t> leaf_cmp(num_cmp * num_digits, 0);
  std::vector<uint8_t> leaf_eq(num_cmp * num_digits, 0);

  if (is_sender_) {
    emp::PRG prg;
    prg.random_bool(reinterpret_cast<bool*>(leaf_cmp.data()), leaf_cmp.size());
    prg.random_bool(reinterpret_cast<bool*>(leaf_eq.data()), leaf_eq.size());

    // n*M instances of 1-of-N OT
    std::vector<uint8_t> leaf_ot_msg(radix * num_cmp * num_digits, 0);
    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg(num_cmp * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] = {leaf_ot_msg.data() + i * radix, radix};
    }

    for (size_t i = 0; i < num_cmp; ++i) {
      auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
      auto* this_digit = digits.data() + i * num_digits;
      auto* this_leaf_cmp = leaf_cmp.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;

      // Step 6, 7 of Alg1 in CF2's paper
      for (size_t j = 0; j < num_digits; ++j) {
        uint8_t rnd_cmp = this_leaf_cmp[j] & 1;
        uint8_t rnd_eq = this_leaf_eq[j] & 1;
        SetLeafOTMsg(this_ot_msg[j], this_digit[j], rnd_cmp, rnd_eq,
                     greater_than);
      }
    }

    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg), radix,
                                             /*bitwidth*/ 2);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(digits), radix,
                                               absl::MakeSpan(leaf_cmp), 2);
    // extract equality bits from packed messages
    for (size_t i = 0; i < num_cmp; ++i) {
      auto* this_leaf_cmp = leaf_cmp.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;
      for (size_t j = 0; j < num_digits; ++j) {
        this_leaf_eq[j] = (this_leaf_cmp[j] >> 1) & 1;
        this_leaf_cmp[j] &= 1;
      }
    }
  }

  using BShrTy = semi2k::BShrTy;
  auto boolean_t = makeType<BShrTy>(field, 1);

  ArrayRef outp(makeType<RingTy>(field), numel);
  if (keep_eq != nullptr) {
    *keep_eq = ring_zeros(field, numel);
  }

  ArrayRef prev_cmp = ring_zeros(field, num_digits * numel).as(boolean_t);
  ArrayRef prev_eq = ring_zeros(field, num_digits * numel).as(boolean_t);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    ArrayView<ring2k_t> xprev_cmp(prev_cmp);
    ArrayView<ring2k_t> xprev_eq(prev_eq);
    // NOTE(juhou): leaf_cmp, and leaf_eq are padded to 8-align
    // But we only take care the first `numel` elements
    pforeach(0, xprev_cmp.numel(), [&](int64_t i) {
      xprev_cmp[i] = leaf_cmp[i];
      xprev_eq[i] = leaf_eq[i];
    });
  });

  // m0[0], m0[1], ..., m0[M],
  // m1[0], m1[1], ..., m1[M],
  // ...
  // mN[0], mN[1], ..., mN[M],
  // slice0: m0[0], m0[2], ..., m0[2*j]
  //         m1[0], m1[2], ..., m1[2*j]
  //
  // slice1: m0[1], m0[3], ..., m0[2*j+1]
  //         m1[1], m0[3], ..., m0[2*j+1]
  size_t current_num_digits = num_digits;
  while (current_num_digits > 1) {
    SPU_ENFORCE((current_num_digits & 1) == 0);
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    // cmp[i-1, j] <- cmp[i,2*j] * eq[i,2*j+1] ^ cmp[i,2*j+1]
    SPU_ENFORCE_EQ(static_cast<size_t>(prev_eq.numel()),
                   numel * current_num_digits);
    auto lhs_eq = prev_eq.slice(0, numel * current_num_digits, 2);
    auto rhs_eq = prev_eq.slice(1, numel * current_num_digits, 2);
    auto next_eq = basic_ot_prot_->BitwiseAnd(lhs_eq, rhs_eq);

    auto lhs_cmp = prev_cmp.slice(0, numel * current_num_digits, 2);
    auto rhs_cmp = prev_cmp.slice(1, numel * current_num_digits, 2);
    auto tmp_cmp = basic_ot_prot_->BitwiseAnd(lhs_cmp, rhs_eq);
    auto next_cmp = ring_xor(rhs_cmp, tmp_cmp);

    prev_eq = next_eq;
    prev_cmp = next_cmp;
    current_num_digits = prev_eq.numel() / numel;
  }

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    ArrayView<const ring2k_t> xcmp(prev_cmp);
    ArrayView<ring2k_t> xout(outp);
    SPU_ENFORCE_EQ(xcmp.numel(), (int64_t)numel);
    pforeach(0, numel, [&](int64_t i) { xout[i] = xcmp[i]; });
    outp = outp.as(boolean_t);

    if (keep_eq) {
      *keep_eq = ring_zeros(field, numel);
      ArrayView<const ring2k_t> xeq(prev_eq);
      ArrayView<ring2k_t> xeq_out(*keep_eq);
      pforeach(0, numel, [&](int64_t i) { xeq_out[i] = xeq[i]; });
      *keep_eq = keep_eq->as(boolean_t);
    }
  });

  return outp;
}

ArrayRef CompareProtocol::Compute(const ArrayRef& inp, bool greater_than) {
  return DoCompute(inp, greater_than, nullptr);
}

std::array<ArrayRef, 2> CompareProtocol::ComputeWithEq(const ArrayRef& inp,
                                                       bool greater_than) {
  ArrayRef eq;
  auto cmp = DoCompute(inp, greater_than, &eq);
  return {cmp, eq};
}

}  // namespace spu::mpc::cheetah
