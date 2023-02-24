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

CompareProtocol::CompareProtocol(std::shared_ptr<BasicOTProtocols> base,
                                 size_t compare_radix)
    : compare_radix_(compare_radix), basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
  SPU_ENFORCE(compare_radix_ >= 1 && compare_radix_ <= 8);
  is_sender_ = base->Rank() == 0;
}

CompareProtocol::~CompareProtocol() { basic_ot_prot_->Flush(); }

void SetLeafOTMsg(absl::Span<uint8_t> ot_messages, uint8_t digit,
                  uint8_t rnd_cmp_bit, uint8_t rnd_eq_bit, bool gt) {
  size_t N = ot_messages.size();
  SPU_ENFORCE(digit <= N, fmt::format("N={} got digit={}", N, digit));
  for (size_t i = 0; i < N; i++) {
    if (gt) {
      ot_messages[i] = rnd_cmp_bit ^ static_cast<uint8_t>(digit > i);
    } else {
      ot_messages[i] = rnd_cmp_bit ^ static_cast<uint8_t>(digit < i);
    }

    // compact two bits into one OT message
    ot_messages[i] |= ((rnd_eq_bit ^ static_cast<uint8_t>(digit == i)) << 1);
  }
}

// The Mill protocol from "CrypTFlow2: Practical 2-Party Secure Inference"
// Algorithm 1. REF: https://arxiv.org/pdf/2010.06457.pdf
ArrayRef CompareProtocol::DoCompute(const ArrayRef& inp, bool greater_than,
                                    ArrayRef* keep_eq) {
  auto field = inp.eltype().as<Ring2k>()->field();
  size_t bit_width = SizeOf(field) * 8;
  SPU_ENFORCE(bit_width % compare_radix_ == 0, "invalid compare radix {}",
              compare_radix_);

  size_t num_digits = CeilDiv(bit_width, compare_radix_);
  size_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
  size_t num_cmp = inp.numel();
  // init to all zero
  std::vector<uint8_t> digits(num_cmp * num_digits, 0);

  // Step 1 break into digits \in [0, radix)
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    ArrayView<u2k> xinp(inp);

    for (size_t i = 0; i < num_cmp; ++i) {
      for (size_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * compare_radix_;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
      }
    }
  });

  std::vector<uint8_t> leaf_cmp(num_cmp * num_digits, 0);
  std::vector<uint8_t> leaf_eq(num_cmp * num_digits, 0);
  if (is_sender_) {
    // Step 2 sample random bits
    emp::PRG prg;
    prg.random_bool(reinterpret_cast<bool*>(leaf_cmp.data()), leaf_cmp.size());
    prg.random_bool(reinterpret_cast<bool*>(leaf_eq.data()), leaf_eq.size());

    // Step 6-7 set the OT messages with two packed bits (one for compare, one
    // for equal)
    std::vector<uint8_t> leaf_ot_msg(radix * num_cmp * num_digits, 0);
    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg(num_cmp * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] =
          absl::Span<uint8_t>{leaf_ot_msg.data() + i * radix, radix};
    }

    for (size_t i = 0; i < num_cmp; ++i) {
      auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
      auto* this_digit = digits.data() + i * num_digits;
      auto* this_leaf_cmp = leaf_cmp.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;

      for (size_t j = 0; j < num_digits; ++j) {
        uint8_t rnd_cmp = this_leaf_cmp[j] & 1;
        uint8_t rnd_eq = this_leaf_eq[j] & 1;
        SetLeafOTMsg(this_ot_msg[j], this_digit[j], rnd_cmp, rnd_eq,
                     greater_than);
      }
    }

    // Step 9: sender of n*M instances of 1-of-N OT
    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg), radix,
                                             /*bitwidth*/ 2);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    // Step 10: receiver of 1-of-N OT
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
  ArrayRef prev_cmp = ring_zeros(field, num_digits * num_cmp).as(boolean_t);
  ArrayRef prev_eq = ring_zeros(field, num_digits * num_cmp).as(boolean_t);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    ArrayView<ring2k_t> xprev_cmp(prev_cmp);
    ArrayView<ring2k_t> xprev_eq(prev_eq);
    pforeach(0, xprev_cmp.numel(), [&](int64_t i) {
      xprev_cmp[i] = leaf_cmp[i];
      xprev_eq[i] = leaf_eq[i];
    });
  });

  // Step 12 - 17. Evaluate the traversal AND
  if (keep_eq == nullptr) {
    // Optimization when keep_eq is false
    // Section 3.1.1: "Removing unnecessary equality computations"
    return TraversalAND(prev_cmp, prev_eq, num_cmp, num_digits).as(boolean_t);
  }

  auto [_gt, _eq] = TraversalANDWithEq(prev_cmp, prev_eq, num_cmp, num_digits);
  *keep_eq = _eq;
  keep_eq->as(boolean_t);
  return _gt.as(boolean_t);
}

std::array<ArrayRef, 2> CompareProtocol::TraversalANDWithEq(ArrayRef cmp,
                                                            ArrayRef eq,
                                                            size_t num_input,
                                                            size_t num_digits) {
  SPU_ENFORCE_EQ(cmp.numel(), eq.numel());
  SPU_ENFORCE_EQ(num_input * num_digits, (size_t)cmp.numel());

  for (size_t i = 1; i <= num_digits; i += 1) {
    size_t current_num_digits = num_digits / (1 << (i - 1));
    if (current_num_digits == 1) break;
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    // cmp[i-1, j] <- cmp[i,2*j] * eq[i,2*j+1] ^ cmp[i,2*j+1]
    size_t n = current_num_digits * num_input;
    auto lhs_eq = eq.slice(0, n, 2);
    auto rhs_eq = eq.slice(1, n, 2);

    auto lhs_cmp = cmp.slice(0, n, 2);
    auto rhs_cmp = cmp.slice(1, n, 2);

    // Correlated ANDs
    //   _eq = rhs_eq & lhs_eq
    //  _cmp = rhs_eq & lhs_cmp
    auto [_eq, _cmp] =
        basic_ot_prot_->CorrelatedBitwiseAnd(rhs_eq, lhs_eq, lhs_cmp);
    eq = _eq;
    cmp = ring_xor(_cmp, rhs_cmp);
  }

  return {cmp, eq};
}

ArrayRef CompareProtocol::TraversalAND(ArrayRef cmp, ArrayRef eq,
                                       size_t num_input, size_t num_digits) {
  // Tree-based traversal ANDs
  // lt0[0], lt0[1], ..., lt0[M],
  // lt1[0], lt1[1], ..., lt1[M],
  // ...
  // ltN[0], ltN[1], ..., ltN[M],
  //
  // View two slices as Nx(M/2) matrix
  // Each row contains M/2 digits.
  // Slice0 contains the even digits
  // slice0: lt0[0], lt0[2], ..., lt0[2*j]
  //         lt1[0], lt1[2], ..., lt1[2*j]
  //         ....
  //         ltn[0], ltn[2], ..., ltn[2*j]
  //
  // Slice1 contains the odd digits
  // slice1: lt0[1], lt0[3], ..., lt0[2*j+1]
  //         lt1[1], lt0[3], ..., lt0[2*j+1]
  //         ....
  //         ltn[1], ltn[3], ..., ltn[2*j+1]
  SPU_ENFORCE_EQ(cmp.numel(), eq.numel());
  SPU_ENFORCE_EQ(num_input * num_digits, (size_t)cmp.numel());

  for (size_t i = 1; i <= num_digits; i += 1) {
    size_t current_num_digits = num_digits / (1 << (i - 1));
    if (current_num_digits == 1) break;
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    // cmp[i-1, j] <- cmp[i,2*j] * eq[i,2*j+1] ^ cmp[i,2*j+1]
    size_t n = current_num_digits * num_input;
    auto lhs_eq = eq.slice(0, n, 2);
    auto rhs_eq = eq.slice(1, n, 2);
    auto lhs_cmp = cmp.slice(0, n, 2);
    auto rhs_cmp = cmp.slice(1, n, 2);

    if (current_num_digits == 2) {
      cmp = basic_ot_prot_->BitwiseAnd(lhs_cmp, rhs_eq);
      ring_xor_(cmp, rhs_cmp);
      // We skip the ANDs for eq on the last loop
      continue;
    }

    // We skip the AND on the 0-th digit which is unnecessary for the next loop.
    size_t nrow = num_input;
    size_t ncol = current_num_digits / 2;
    SPU_ENFORCE_EQ((size_t)lhs_eq.numel(), nrow * ncol);

    ArrayRef _lhs_eq(lhs_eq.eltype(), nrow * (ncol - 1));
    ArrayRef _rhs_eq(lhs_eq.eltype(), nrow * (ncol - 1));
    ArrayRef _lhs_cmp(lhs_cmp.eltype(), nrow * (ncol - 1));

    ArrayRef _lhs_cmp_col0(lhs_cmp.eltype(), nrow);
    ArrayRef _rhs_eq_col0(rhs_eq.eltype(), nrow);

    for (size_t r = 0; r < nrow; ++r) {
      std::memcpy(&_rhs_eq_col0.at(r), &rhs_eq.at(r * ncol), rhs_eq.elsize());
      std::memcpy(&_lhs_cmp_col0.at(r), &lhs_cmp.at(r * ncol),
                  lhs_cmp.elsize());

      for (size_t c = 1; c < ncol; ++c) {
        std::memcpy(&_lhs_cmp.at(r * (ncol - 1) + c - 1),
                    &lhs_cmp.at(r * ncol + c), lhs_eq.elsize());
        std::memcpy(&_lhs_eq.at(r * (ncol - 1) + c - 1),
                    &lhs_eq.at(r * ncol + c), lhs_eq.elsize());
        std::memcpy(&_rhs_eq.at(r * (ncol - 1) + c - 1),
                    &rhs_eq.at(r * ncol + c), rhs_eq.elsize());
      }
    }
    // Normal AND on 0-th column
    auto _next_cmp_col0 =
        basic_ot_prot_->BitwiseAnd(_rhs_eq_col0, _lhs_cmp_col0);

    // Correlated AND on the remain columns
    auto [_next_cmp, _next_eq] =
        basic_ot_prot_->CorrelatedBitwiseAnd(_rhs_eq, _lhs_cmp, _lhs_eq);

    eq = ArrayRef(_next_eq.eltype(), nrow * ncol);
    cmp = ArrayRef(_next_cmp.eltype(), nrow * ncol);

    for (size_t r = 0; r < nrow; ++r) {
      std::memcpy(&cmp.at(r * ncol), &_next_cmp_col0.at(r), cmp.elsize());

      for (size_t c = 1; c < ncol; ++c) {
        std::memcpy(&cmp.at(r * ncol + c),
                    &_next_cmp.at(r * (ncol - 1) + c - 1), _next_cmp.elsize());

        std::memcpy(&eq.at(r * ncol + c), &_next_eq.at(r * (ncol - 1) + c - 1),
                    _next_eq.elsize());
      }
    }

    ring_xor_(cmp, rhs_cmp);
  }

  return cmp;
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
