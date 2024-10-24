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
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/cheetah/yacl_ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/yacl_ot/util.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

TruncateProtocol::TruncateProtocol(std::shared_ptr<BasicOTProtocols> base)
    : basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
}

TruncateProtocol::~TruncateProtocol() { basic_ot_prot_->Flush(); }

NdArrayRef TruncateProtocol::ComputeWrap(const NdArrayRef& inp,
                                         const Meta& meta) {
  const int rank = basic_ot_prot_->Rank();

  switch (meta.sign) {
    case SignType::Positive: {
      if (!meta.signed_arith) {
        // without sign flip
        return MSB0ToWrap(inp, meta.shift_bits);
      } else {
        // MSB=0 with sign flip equals to MSB=1
        return MSB1ToWrap(inp, meta.shift_bits);
      }
      break;
    }
    case SignType::Negative: {
      if (!meta.signed_arith) {
        // without sign flip
        return MSB1ToWrap(inp, meta.shift_bits);
      } else {
        // MSB=1 with sign flip equals to MSB=0
        return MSB0ToWrap(inp, meta.shift_bits);
      }
      break;
    }
    case SignType::Unknown:
    default: {
      CompareProtocol compare_prot(basic_ot_prot_);
      NdArrayRef wrap_bool;
      // w = 1{x_A + x_B > 2^k - 1}
      //   = 1{x_A > 2^k - 1 - x_B}
      const auto field = inp.eltype().as<Ring2k>()->field();
      if (rank == 0) {
        wrap_bool = compare_prot.Compute(inp, true);
      } else {
        auto adjusted = ring_neg(inp);
        DISPATCH_ALL_FIELDS(field, "wrap_adjust", [&]() {
          NdArrayView<ring2k_t> xadj(adjusted);
          pforeach(0, inp.numel(), [&](int64_t i) { xadj[i] -= 1; });
        });
        wrap_bool = compare_prot.Compute(adjusted, true);
      }
      return basic_ot_prot_->B2ASingleBitWithSize(
          wrap_bool.as(makeType<BShrTy>(field, 1)), meta.shift_bits);
      break;
    }
  }
}

// Given msb(xA + xB mod 2^k) = 1, and xA, xB \in [0, 2^k)
// To compute w = 1{xA + xB > 2^{k} - 1}.
//            w = msb(xA) & msb(xB).
// COT msg corr=msb(xA) on choice msb(xB)
//    - msb(xB) = 0: get(-x, x) => 0
//    - msb(xB) = 1: get(-x, x + msb(xA)) => msb(xA)
NdArrayRef TruncateProtocol::MSB1ToWrap(const NdArrayRef& inp,
                                        size_t shift_bits) {
  const auto field = inp.eltype().as<Ring2k>()->field();
  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const size_t bw = SizeOf(field) * 8;

  NdArrayRef cot_output = ring_zeros(field, inp.shape());
  DISPATCH_ALL_FIELDS(field, "MSB1ToWrap", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    NdArrayView<const u2k> xinp(inp);
    auto xout = absl::MakeSpan(&cot_output.at<u2k>(0), cot_output.numel());

    if (rank == 0) {
      std::vector<u2k> cot_input(numel);
      pforeach(0, numel,
               [&](int64_t i) { cot_input[i] = ((xinp[i] >> (bw - 1)) & 1); });

      auto sender = basic_ot_prot_->GetSenderCOT();
      sender->SendCAMCC(absl::MakeSpan(cot_input), xout, shift_bits);
      sender->Flush();
      pforeach(0, numel, [&](int64_t i) { xout[i] = -xout[i]; });
    } else {
      std::vector<uint8_t> cot_input(numel);
      pforeach(0, numel,
               [&](int64_t i) { cot_input[i] = ((xinp[i] >> (bw - 1)) & 1); });

      basic_ot_prot_->GetReceiverCOT()->RecvCAMCC(absl::MakeSpan(cot_input),
                                                  xout, shift_bits);
    }
  });

  return cot_output.as(makeType<BShrTy>(field, 1));
}

// Given msb(xA + xB mod 2^k) = 0, and xA, xB \in [0, 2^k)
// To compute w = 1{xA + xB > 2^{k} - 1}.
//
// Given msb(xA + xB mod 2^k) = 0
//   1. when xA + xB = x => w = 0
//   2. when xA + xB = x + 2^{k} => w = 1
//   For case 1: msb(xA) = msb(xB) = 0 or msb(xA) = msb(xB) = 1
//   For case 2: msb(xA) = 1 or msb(xB) = 1.
// Thus w = msb(xA) | msb(xB)
//
// 1-of-2 OT msg (r^msb(xA), r^1) on choice msb(xB)
//   - msb(xB) = 0: get (r, r^msb(xA)) => msb(xA)
//   - msb(xB) = 1: get (r, r^1) => 1
NdArrayRef TruncateProtocol::MSB0ToWrap(const NdArrayRef& inp,
                                        size_t shift_bits) {
  const auto field = inp.eltype().as<Ring2k>()->field();
  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const size_t bw = SizeOf(field) * 8;

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  NdArrayRef outp;
  if (0 == rank) {
    outp = ring_randbit(field, inp.shape());
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(field, "MSB0_adjust", [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const u2k> xinp(inp);
      NdArrayView<const u2k> xrnd(outp);
      // when msb(xA) = 0, set (r, 1^r)
      //  ow. msb(xA) = 1, set (1^r, 1^r)
      // Equals to (r^msb(xA), r^1)
      for (int64_t i = 0; i < numel; ++i) {
        send[2 * i + 0] = xrnd[i] ^ ((xinp[i] >> (bw - 1)) & 1);
        send[2 * i + 1] = xrnd[i] ^ 1;
      }
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);
    DISPATCH_ALL_FIELDS(field, "MSB0_adjust", [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const u2k> xinp(inp);
      for (int64_t i = 0; i < numel; ++i) {
        choices[i] = (xinp[i] >> (bw - 1)) & 1;
      }
    });

    std::vector<uint8_t> recv(numel);
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                               absl::MakeSpan(recv), nbits);

    outp = ring_zeros(field, inp.shape());
    DISPATCH_ALL_FIELDS(field, "MSB0_finalize", [&]() {
      NdArrayView<ring2k_t> xoup(outp);
      pforeach(0, numel, [&](int64_t i) {
        xoup[i] = static_cast<ring2k_t>(recv[i] & 1);
      });
    });
  }

  return basic_ot_prot_->B2ASingleBitWithSize(
      outp.as(makeType<BShrTy>(field, 1)), static_cast<int>(shift_bits));
}

NdArrayRef TruncateProtocol::Compute(const NdArrayRef& inp, Meta meta) {
  size_t shift = meta.shift_bits;
  if (shift == 0) {
    return inp;
  }

  auto field = inp.eltype().as<Ring2k>()->field();
  const size_t bit_width = SizeOf(field) * 8;
  SPU_ENFORCE(shift < bit_width, "truncate should not truncate full bit width");
  if (meta.signed_arith) {
    SPU_ENFORCE((bit_width >= shift + 1),
                "signed truncate should keep the sign bit");
  }

  const int rank = basic_ot_prot_->Rank();

  if (meta.signed_arith && meta.sign == SignType::Unknown &&
      meta.use_heuristic) {
    // Use heuristic optimization from SecureQ8: Add a large positive to make
    // sure the value is always positive
    // We assume |x| < 2^{k - b - 1}
    // 1. x' = x + 2^{k - b} (should no wrap round 2^k)
    // 2. y = TruncMSB0(x' ,f) ie y = (x + 2^{k - b}) / 2^f
    // 3. output y - 2^{k - b - f}
    meta.use_heuristic = false;
    meta.sign = SignType::Positive;

    if (rank == 0) {
      NdArrayRef tmp = inp.clone();
      DISPATCH_ALL_FIELDS(field, "trunc_with_heuristic", [&] {
        NdArrayView<ring2k_t> _inp(tmp);
        ring2k_t big_value = static_cast<ring2k_t>(1)
                             << (bit_width - kHeuristicBound);
        pforeach(0, inp.numel(),
                 [&](int64_t i) { _inp[i] = _inp[i] + big_value; });
      });

      tmp = Compute(tmp, meta);

      DISPATCH_ALL_FIELDS(field, "trunc_with_heuristic", [&] {
        NdArrayView<ring2k_t> _outp(tmp);
        ring2k_t big_value = static_cast<ring2k_t>(1)
                             << (bit_width - kHeuristicBound - shift);
        pforeach(0, inp.numel(),
                 [&](int64_t i) { _outp[i] = _outp[i] - big_value; });
      });
      return tmp;
    } else {
      return Compute(inp, meta);
    }
  }

  NdArrayRef wrap_ashr;
  NdArrayRef out = ring_zeros(field, inp.shape());

  return DISPATCH_ALL_FIELDS(field, "Truncate", [&]() {
    const ring2k_t component = (static_cast<ring2k_t>(1) << (bit_width - 1));
    NdArrayView<const ring2k_t> xinp(inp);

    // Compute w = 1{x0 + x1 >= 2^{k}}
    if (meta.signed_arith && rank == 0) {
      // For signed arith right shift, we convert to unsigned logic right shift
      // by convert to two-component form.
      auto tmp = ring_zeros(field, inp.shape());
      NdArrayView<ring2k_t> xtmp(tmp);
      pforeach(0, inp.numel(),
               [&](int64_t i) { xtmp[i] = xinp[i] + component; });
      wrap_ashr = ComputeWrap(tmp, meta);
    } else {
      wrap_ashr = ComputeWrap(inp, meta);
    }
    NdArrayView<const ring2k_t> xwrap(wrap_ashr);

    // NOTE(lwj) We need logic right shift here
    /// m' = (m >> shift) - wrap * 2^{k - shift}
    // [m']_A = (m0 >> shift) - [wrap]_A * 2^{k - shift}
    NdArrayView<ring2k_t> xout(out);
    if (meta.signed_arith && rank == 0) {
      pforeach(0, inp.numel(), [&](int64_t i) {
        xout[i] = ((xinp[i] + component) >> shift);
        xout[i] -= (xwrap[i] << (bit_width - shift));
      });
    } else {
      pforeach(0, inp.numel(), [&](int64_t i) {
        xout[i] = (xinp[i] >> shift) - (xwrap[i] << (bit_width - shift));
      });
    }

    if (meta.signed_arith && rank == 0) {
      ring2k_t u = static_cast<ring2k_t>(1) << (bit_width - shift - 1);
      pforeach(0, inp.numel(), [&](int64_t i) { xout[i] -= u; });
    }

    if (rank == 0) {
      // The origin Truncate introduce -1 error by 50%.
      // We balance it by +1 at the rate of 50%.
      // As a result, we introduce 0 error by 50%， +1 error by 25% and -1 error
      // by 25%.
      pforeach(0, inp.numel(), [&](int64_t i) { xout[i] += (xout[i] & 1); });
    }

    basic_ot_prot_->Flush();
    return out.as(inp.eltype());
  });
}

}  // namespace spu::mpc::cheetah
