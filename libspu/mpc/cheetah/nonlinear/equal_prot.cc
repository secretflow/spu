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

#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

EqualProtocol::EqualProtocol(const std::shared_ptr<BasicOTProtocols>& base,
                             size_t compare_radix)
    : compare_radix_(compare_radix), basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
  SPU_ENFORCE(compare_radix_ >= 1 && compare_radix_ <= 8);
  is_sender_ = base->Rank() == 0;
}

EqualProtocol::~EqualProtocol() { basic_ot_prot_->Flush(); }

static void SetLeafOTMsg(absl::Span<uint8_t> ot_messages, uint8_t digit,
                         uint8_t rnd_eq_bit) {
  size_t N = ot_messages.size();
  SPU_ENFORCE(digit <= N, fmt::format("N={} got digit={}", N, digit));
  std::fill_n(ot_messages.data(), N, rnd_eq_bit);
  for (size_t i = 0; i < N; i++) {
    ot_messages[i] = rnd_eq_bit ^ static_cast<uint8_t>(digit == i);
  }
}

NdArrayRef EqualProtocol::DoCompute(const NdArrayRef& inp, size_t bit_width) {
  auto field = inp.eltype().as<Ring2k>()->field();
  if (bit_width == 0) {
    bit_width = SizeOf(field) * 8;
  }
  bit_width = std::min(bit_width, SizeOf(field) * 8);

  int64_t remain = bit_width % compare_radix_;
  int64_t num_digits = CeilDiv(bit_width, compare_radix_);
  int64_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
  int64_t num_cmp = inp.numel();
  // init to all zero
  std::vector<uint8_t> digits(num_cmp * num_digits, 0);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    const auto mask_remain = makeBitsMask<u2k>(remain);
    NdArrayView<u2k> xinp(inp);

    for (int64_t i = 0; i < num_cmp; ++i) {
      for (int64_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * compare_radix_;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
        // last digits
        if (remain > 0 && (j + 1 == num_digits)) {
          digits[i * num_digits + j] &= mask_remain;
        }
      }
    }
  });

  std::vector<uint8_t> leaf_eq(num_cmp * num_digits, 0);
  if (is_sender_) {
    yacl::crypto::Prg<uint8_t> prg(yacl::crypto::SecureRandSeed());
    prg.Fill(absl::MakeSpan(leaf_eq));
    // convert u8 random to boolean random
    std::transform(leaf_eq.begin(), leaf_eq.end(), leaf_eq.data(),
                   [](uint8_t v) { return v & 1; });

    // n*M instances of 1-of-N OT
    std::vector<uint8_t> leaf_ot_msg(radix * num_cmp * num_digits, 0);

    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg(num_cmp * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] =
          absl::MakeSpan(leaf_ot_msg.data() + i * radix, radix);
    }

    for (int64_t i = 0; i < num_cmp; ++i) {
      auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
      auto* this_digit = digits.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;

      // Step 6, 7 of Alg1 in CF2's paper
      for (int64_t j = 0; j < num_digits; ++j) {
        uint8_t rnd_eq = this_leaf_eq[j] & 1;
        SetLeafOTMsg(this_ot_msg[j], this_digit[j], rnd_eq);
      }
    }

    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg), radix,
                                             /*bitwidth*/ 1);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(digits), radix,
                                               absl::MakeSpan(leaf_eq), 1);
  }

  auto boolean_t = makeType<BShrTy>(field, 1);
  NdArrayRef prev_eq =
      ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
          .as(boolean_t);

  // Transpose from msg-major order
  // m0[0], m0[1], ..., m0[M],
  // m1[0], m1[1], ..., m1[M],
  // ...
  // mN[0], mN[1], ..., mN[M],
  //
  // To digit-major order
  //
  // m0[0], m1[0], ..., mN[0]
  // m0[1], m1[1], ..., mN[1]
  // ...
  // m0[M], m1[M], ..., mN[M]
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> xprev_eq(prev_eq);
    for (int64_t r = 0; r < num_cmp; ++r) {
      for (int64_t c = 0; c < num_digits; ++c) {
        xprev_eq[c * num_cmp + r] = leaf_eq[r * num_digits + c];
      }
    }
  });

  // Tree-based traversal ANDs
  // m0[0], m1[0], ..., mN[0]
  // m0[1], m1[1], ..., mN[1]
  // ...
  // m0[M], m1[M], ..., mN[M]
  //     ||
  //     \/
  // eq[0], eq[1], ..., eq[N]
  int64_t current_num_digits = num_digits;
  while (current_num_digits > 1) {
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    int64_t half_d = current_num_digits / 2;
    auto lhs_eq = prev_eq.slice({0}, {half_d * num_cmp}, {1});
    auto rhs_eq = prev_eq.slice({lhs_eq.numel()}, {lhs_eq.numel() * 2}, {1});
    SPU_ENFORCE_EQ(lhs_eq.numel(), rhs_eq.numel());

    int64_t remain_d = current_num_digits - 2 * half_d;
    auto next_eq = basic_ot_prot_->BitwiseAnd(lhs_eq, rhs_eq);
    if (remain_d > 0) {
      NdArrayRef tmp(prev_eq.eltype(), {next_eq.numel() + remain_d * num_cmp});

      std::memcpy(&tmp.at(0), &next_eq.at(0),
                  prev_eq.elsize() * next_eq.numel());

      std::memcpy(&tmp.at(next_eq.numel()), &prev_eq.at(half_d * 2 * num_cmp),
                  prev_eq.elsize() * remain_d * num_cmp);
      prev_eq = tmp;
    } else {
      prev_eq = next_eq;
    }
    current_num_digits = CeilDiv(prev_eq.numel(), num_cmp);
  }

  return prev_eq.reshape(inp.shape()).as(boolean_t);
}

NdArrayRef EqualProtocol::Compute(const NdArrayRef& inp, size_t bit_width) {
  return DoCompute(inp, bit_width);
}

}  // namespace spu::mpc::cheetah
