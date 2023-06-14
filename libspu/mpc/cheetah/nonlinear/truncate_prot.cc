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
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/util.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

TruncateProtocol::TruncateProtocol(std::shared_ptr<BasicOTProtocols> base)
    : basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
}

TruncateProtocol::~TruncateProtocol() { basic_ot_prot_->Flush(); }

ArrayRef TruncateProtocol::ComputeWrap(const ArrayRef& inp, const Meta& meta) {
  const int rank = basic_ot_prot_->Rank();

  switch (meta.msb) {
    case MSB_st::zero: {
      if (!meta.signed_arith) {
        // without sign flip
        return MSB0ToWrap(inp, meta.shift_bits);
      } else {
        // MSB=0 with sign flip equals to MSB=1
        return MSB1ToWrap(inp, meta.shift_bits);
      }
      break;
    }
    case MSB_st::one: {
      if (!meta.signed_arith) {
        // without sign flip
        return MSB1ToWrap(inp, meta.shift_bits);
      } else {
        // MSB=1 with sign flip equals to MSB=0
        return MSB0ToWrap(inp, meta.shift_bits);
      }
      break;
    }
    case MSB_st::unknown:
    default: {
      CompareProtocol compare_prot(basic_ot_prot_);
      ArrayRef wrap_bool;
      // w = 1{x_A + x_B > 2^k - 1}
      //   = 1{x_A > 2^k - 1 - x_B}
      const auto field = inp.eltype().as<Ring2k>()->field();
      if (rank == 0) {
        wrap_bool = compare_prot.Compute(inp, true);
      } else {
        auto adjusted = ring_neg(inp);
        DISPATCH_ALL_FIELDS(field, "", [&]() {
          ArrayView<ring2k_t> xadj(adjusted);
          pforeach(0, inp.numel(), [&](int64_t i) { xadj[i] -= 1; });
        });
        wrap_bool = compare_prot.Compute(adjusted, true);
      }
      return basic_ot_prot_->B2ASingleBitWithSize(
          wrap_bool.as(makeType<semi2k::BShrTy>(field, 1)), meta.shift_bits);
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
ArrayRef TruncateProtocol::MSB1ToWrap(const ArrayRef& inp, size_t shift_bits) {
  const auto field = inp.eltype().as<Ring2k>()->field();
  const size_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const size_t bw = SizeOf(field) * 8;

  ArrayRef cot_output = ring_zeros(field, numel);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    ArrayView<const u2k> xinp(inp);
    ArrayView<u2k> xout(cot_output);

    if (rank == 0) {
      std::vector<u2k> cot_input(numel);
      pforeach(0, numel,
               [&](int64_t i) { cot_input[i] = ((xinp[i] >> (bw - 1)) & 1); });

      auto sender = basic_ot_prot_->GetSenderCOT();
      sender->SendCAMCC(absl::MakeSpan(cot_input),
                        {xout.data(), (size_t)xout.numel()}, shift_bits);
      sender->Flush();
      pforeach(0, numel, [&](int64_t i) { xout[i] = -xout[i]; });
    } else {
      std::vector<uint8_t> cot_input(numel);
      pforeach(0, numel,
               [&](int64_t i) { cot_input[i] = ((xinp[i] >> (bw - 1)) & 1); });

      basic_ot_prot_->GetReceiverCOT()->RecvCAMCC(
          absl::MakeSpan(cot_input), {xout.data(), (size_t)xout.numel()},
          shift_bits);
    }
  });

  return cot_output.as(makeType<semi2k::BShrTy>(field, 1));
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
ArrayRef TruncateProtocol::MSB0ToWrap(const ArrayRef& inp, size_t shift_bits) {
  const auto field = inp.eltype().as<Ring2k>()->field();
  const size_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const size_t bw = SizeOf(field) * 8;

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  ArrayRef outp;
  if (0 == rank) {
    outp = ring_randbit(field, numel);
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      ArrayView<const u2k> xinp(inp);
      ArrayView<const u2k> xrnd(outp);
      // when msb(xA) = 0, set (r, 1^r)
      //  ow. msb(xA) = 1, set (1^r, 1^r)
      // Equals to (r^msb(xA), r^1)
      for (size_t i = 0; i < numel; ++i) {
        send[2 * i + 0] = xrnd[i] ^ ((xinp[i] >> (bw - 1)) & 1);
        send[2 * i + 1] = xrnd[i] ^ 1;
      }
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      ArrayView<const u2k> xinp(inp);
      for (size_t i = 0; i < numel; ++i) {
        choices[i] = (xinp[i] >> (bw - 1)) & 1;
      }
    });

    std::vector<uint8_t> recv(numel);
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                               absl::MakeSpan(recv), nbits);

    outp = ring_zeros(field, numel);
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      ArrayView<ring2k_t> xoup(outp);
      pforeach(0, numel, [&](int64_t i) {
        xoup[i] = static_cast<ring2k_t>(recv[i] & 1);
      });
    });
  }

  return basic_ot_prot_->B2ASingleBitWithSize(
      outp.as(makeType<semi2k::BShrTy>(field, 1)), (int)shift_bits);
}

ArrayRef TruncateProtocol::Compute(const ArrayRef& inp, Meta meta) {
  size_t shift = meta.shift_bits;
  if (shift == 0) return inp;
  auto field = inp.eltype().as<Ring2k>()->field();
  const size_t bit_width = SizeOf(field) * 8;
  SPU_ENFORCE(shift < bit_width, "truncate should not truncate full bit width");
  if (meta.signed_arith) {
    SPU_ENFORCE((bit_width >= shift + 1),
                "signed truncate should keep the sign bit");
  }

  const int rank = basic_ot_prot_->Rank();

  ArrayRef wrap_ashr;
  ArrayRef out = ring_zeros(field, inp.numel());

  return DISPATCH_ALL_FIELDS(field, "", [&]() {
    const ring2k_t component = (static_cast<ring2k_t>(1) << (bit_width - 1));
    ArrayView<const ring2k_t> xinp(inp);

    // Compute w = 1{x0 + x1 >= 2^{k}}
    if (meta.signed_arith && rank == 0) {
      // For signed arith right shift, we convert to unsigned logic right shift
      // by convert to two-component form.
      auto tmp = ring_zeros(field, inp.numel());
      ArrayView<ring2k_t> xtmp(tmp);
      pforeach(0, inp.numel(),
               [&](int64_t i) { xtmp[i] = xinp[i] + component; });
      wrap_ashr = ComputeWrap(tmp, meta);
    } else {
      wrap_ashr = ComputeWrap(inp, meta);
    }
    ArrayView<const ring2k_t> xwrap(wrap_ashr);

    // NOTE(juhou) We need logic right shift here
    /// m' = (m >> shift) - wrap * 2^{k - shift}
    // [m']_A = (m0 >> shift) - [wrap]_A * 2^{k - shift}
    ArrayView<ring2k_t> xout(out);
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

    basic_ot_prot_->Flush();
    return out.as(inp.eltype());
  });
}

}  // namespace spu::mpc::cheetah
