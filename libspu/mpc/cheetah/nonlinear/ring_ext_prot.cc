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

#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"

#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

namespace {
template <typename T>
auto makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

NdArrayRef CastTo(const NdArrayRef &x, FieldType from, FieldType to) {
  SPU_ENFORCE_EQ(x.eltype().as<RingTy>()->field(), from);
  // cast from src type to dst type
  auto out = ring_zeros(to, x.shape());
  DISPATCH_ALL_FIELDS(from, "cast_to", [&]() {
    using sT = ring2k_t;
    NdArrayView<sT> src(x);

    DISPATCH_ALL_FIELDS(to, "cast_to", [&]() {
      using dT = ring2k_t;
      NdArrayView<dT> dst(out);
      pforeach(0, src.numel(),
               [&](int64_t i) { dst[i] = static_cast<dT>(src[i]); });
    });
  });
  return out;
}

}  // namespace

RingExtendProtocol::RingExtendProtocol(
    const std::shared_ptr<BasicOTProtocols> &base)
    : basic_ot_prot_(base) {}

RingExtendProtocol::~RingExtendProtocol() {}

NdArrayRef RingExtendProtocol::UnsignedExtend(const NdArrayRef &inp,
                                              const Meta &meta) {
  SPU_ENFORCE(meta.src_width < meta.dst_width);
  NdArrayRef wrap = ComputeWrap(inp, meta);

  NdArrayRef out = ring_zeros(meta.dst_ring, inp.shape());

  DISPATCH_ALL_FIELDS(meta.src_ring, "zext", [&]() {
    using sT = ring2k_t;
    NdArrayView<const sT> input(inp);

    DISPATCH_ALL_FIELDS(meta.dst_ring, "zext", [&]() {
      using dT = ring2k_t;
      NdArrayView<const dT> w(wrap);
      NdArrayView<dT> output(out);

      auto shift = static_cast<dT>(1) << meta.src_width;
      auto msk0 = (static_cast<dT>(1) << (meta.dst_width - meta.src_width)) - 1;
      auto msk1 = makeMask<dT>(meta.dst_width);

      pforeach(0, input.numel(), [&](int64_t i) {
        output[i] = (static_cast<dT>(input[i]) - shift * (w[i] & msk0)) & msk1;
      });
    });
  });

  return out;
}

// Given msb(xA + xB mod 2^k) = 1, and xA, xB \in [0, 2^k)
// To compute w = 1{xA + xB > 2^{k} - 1}.
//            w = msb(xA) & msb(xB).
// COT msg corr=msb(xA) on choice msb(xB)
//    - msb(xB) = 0: get(-x, x) => 0
//    - msb(xB) = 1: get(-x, x + msb(xA)) => msb(xA)
NdArrayRef RingExtendProtocol::MSB1ToWrap(const NdArrayRef &inp,
                                          const Meta &meta) {
  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const size_t wrap_width = meta.dst_width - meta.src_width;

  NdArrayRef cot_output = ring_zeros(meta.dst_ring, inp.shape());
  DISPATCH_ALL_FIELDS(meta.src_ring, "MSB1ToWrap", [&]() {
    using sT = std::make_unsigned<ring2k_t>::type;
    NdArrayView<const sT> xinp(inp);

    DISPATCH_ALL_FIELDS(meta.dst_ring, "MSB1ToWrap", [&]() {
      using dT = std::make_unsigned<ring2k_t>::type;
      auto xout = absl::MakeSpan(&cot_output.at<dT>(0), cot_output.numel());

      if (rank == 0) {
        std::vector<dT> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = ((xinp[i] >> (meta.src_width - 1)) & 1);
        });

        auto sender = basic_ot_prot_->GetSenderCOT();
        sender->SendCAMCC(absl::MakeSpan(cot_input), xout, wrap_width);
        sender->Flush();

        auto msk = makeMask<dT>(meta.dst_width);
        pforeach(0, numel, [&](int64_t i) { xout[i] = (-xout[i]) & msk; });
      } else {
        std::vector<uint8_t> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = ((xinp[i] >> (meta.src_width - 1)) & 1);
        });

        basic_ot_prot_->GetReceiverCOT()->RecvCAMCC(absl::MakeSpan(cot_input),
                                                    xout, wrap_width);
      }
    });
  });

  return cot_output.as(makeType<BShrTy>(meta.dst_ring, 1));
}

NdArrayRef RingExtendProtocol::MSB0ToWrap(const NdArrayRef &inp,
                                          const Meta &meta) {
  const auto field = inp.eltype().as<Ring2k>()->field();
  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  NdArrayRef wrap_bool;
  if (0 == rank) {
    wrap_bool = ring_randbit(meta.src_ring, inp.shape());
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(field, "MSB0_adjust", [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const u2k> xinp(inp);
      NdArrayView<const u2k> xrnd(wrap_bool);
      // when msb(xA) = 0, set (r, 1^r)
      //  ow. msb(xA) = 1, set (1^r, 1^r)
      // Equals to (r^msb(xA), r^1)
      for (int64_t i = 0; i < numel; ++i) {
        send[2 * i + 0] = xrnd[i] ^ ((xinp[i] >> (meta.src_width - 1)) & 1);
        send[2 * i + 1] = xrnd[i] ^ 1;
      }
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);
    DISPATCH_ALL_FIELDS(meta.src_ring, "MSB0_adjust", [&]() {
      using sT = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const sT> xinp(inp);
      for (int64_t i = 0; i < numel; ++i) {
        choices[i] = (xinp[i] >> (meta.src_width - 1)) & 1;
      }

      std::vector<uint8_t> recv(numel);
      basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                                 absl::MakeSpan(recv), nbits);

      wrap_bool = ring_zeros(meta.src_ring, inp.shape());
      NdArrayView<ring2k_t> xoup(wrap_bool);
      pforeach(0, numel, [&](int64_t i) {
        xoup[i] = static_cast<ring2k_t>(recv[i] & 1);
      });
    });
  }

  // cast from src type to dst type
  NdArrayRef out = CastTo(wrap_bool, meta.src_ring, meta.dst_ring);

  return basic_ot_prot_->B2ASingleBitWithSize(
      out.as(makeType<BShrTy>(meta.dst_ring, 1)),
      static_cast<int>(meta.dst_width - meta.src_width));
}

NdArrayRef RingExtendProtocol::Compute(const NdArrayRef &inp,
                                       const Meta &meta) {
  if (meta.use_heuristic) {
    SPU_ENFORCE(meta.signed_arith, "use_heuristic=true need signed arith=true");
  }
  SPU_ENFORCE(meta.src_width < meta.dst_width);
  SPU_ENFORCE_EQ(meta.src_width, (int64_t)SizeOf(meta.src_ring) * 8,
                 "Now support input from the  defined ring only");
  SPU_ENFORCE(meta.dst_width <= (int64_t)SizeOf(meta.dst_ring) * 8);

  const int rank = basic_ot_prot_->Rank();

  if (meta.signed_arith and meta.sign == SignType::Unknown and
      meta.use_heuristic) {
    // Use heuristic via adding a large positive value to make sure the input
    // is also positive.
    Meta _meta = meta;
    _meta.use_heuristic = false;
    _meta.sign = SignType::Positive;

    if (rank == 0) {
      NdArrayRef tmp = inp.clone();
      DISPATCH_ALL_FIELDS(_meta.src_ring, "ext.adj", [&]() {
        NdArrayView<ring2k_t> tmp_inp(tmp);
        ring2k_t big_val = static_cast<ring2k_t>(1)
                           << (meta.src_width - kHeuristicBound);
        auto msk = makeMask<ring2k_t>(meta.src_width);
        pforeach(0, tmp.numel(),
                 [&](int64_t i) { tmp_inp[i] = (tmp_inp[i] + big_val) & msk; });
      });

      tmp = Compute(tmp, _meta);

      DISPATCH_ALL_FIELDS(_meta.dst_ring, "ext.adj", [&]() {
        NdArrayView<ring2k_t> outp(tmp);
        ring2k_t big_val = static_cast<ring2k_t>(1)
                           << (meta.src_width - kHeuristicBound);
        auto msk = makeMask<ring2k_t>(meta.dst_width);
        pforeach(0, outp.numel(),
                 [&](int64_t i) { outp[i] = (outp[i] - big_val) & msk; });
      });
      return tmp;
    } else {
      // Nothing to do for rank=1
      return Compute(inp, _meta);
    }
  }

  Meta _meta = meta;

  if (meta.signed_arith and meta.sign != SignType::Unknown) {
    // flip sign when dogin signed arithmetic
    // because SignedExt(x, k, k')  = UnsginedExt(x + 2^{k - 1} mod 2^k, k, k')
    // - 2^{k - 1}
    _meta.sign = meta.sign == SignType::Positive ? SignType::Negative
                                                 : SignType::Positive;
  }

  NdArrayRef out = DISPATCH_ALL_FIELDS(meta.src_ring, "ext.zext", [&]() {
    using sT = ring2k_t;
    NdArrayView<const sT> xinp(inp);

    if (_meta.signed_arith and rank == 0) {
      // Compute [x] + 2^{k - 1}
      sT component = static_cast<sT>(1) << (meta.src_width - 1);
      auto msk = makeMask<sT>(meta.src_width);
      auto tmp = ring_zeros(meta.src_ring, inp.shape());
      NdArrayView<sT> xtmp(tmp);
      pforeach(0, tmp.numel(),
               [&](int64_t i) { xtmp[i] = (xinp[i] + component) & msk; });
      return UnsignedExtend(tmp, _meta);
    } else {
      return UnsignedExtend(inp, _meta);
    }
  });

  if (_meta.signed_arith and rank == 0) {
    DISPATCH_ALL_FIELDS(meta.dst_ring, "ext.zext.adj", [&]() {
      using dT = ring2k_t;
      dT component = static_cast<dT>(1) << (meta.src_width - 1);
      auto msk = makeMask<dT>(meta.dst_width);
      NdArrayView<dT> xtmp(out);
      pforeach(0, out.numel(),
               [&](int64_t i) { xtmp[i] = (xtmp[i] - component) & msk; });
    });
  }

  return out;
}

NdArrayRef RingExtendProtocol::ComputeWrap(const NdArrayRef &inp,
                                           const Meta &meta) {
  switch (meta.sign) {
    case SignType::Positive:
      return MSB0ToWrap(inp, meta);
      break;
    case SignType::Negative:
      return MSB1ToWrap(inp, meta);
      break;
    default:
      break;
  }

  CompareProtocol cmp_protocol(basic_ot_prot_);
  NdArrayRef wrap_bool;
  if (basic_ot_prot_->Rank() == 0) {
    wrap_bool = cmp_protocol.Compute(inp, /*gt*/ true);
  } else {
    //     w = 1{x0 + x1 >= 2^k}
    // <=> w = 1{x0 > 2^k - 1 - x1 mod 2^k}
    // <=> w = 1{x0 > -x1 - 1 mod 2^k}
    auto adjusted = ring_neg(inp);  // -x1 mod 2^k
    DISPATCH_ALL_FIELDS(meta.src_ring, "wrap.adj", [&]() {
      NdArrayView<ring2k_t> xadj(adjusted);
      // -1 mod 2^k
      pforeach(0, xadj.numel(), [&](int64_t i) { xadj[i] -= 1; });
    });

    wrap_bool = cmp_protocol.Compute(adjusted, /*gt*/ true);
  }

  auto wrap = CastTo(wrap_bool, meta.src_ring, meta.dst_ring);
  return basic_ot_prot_->B2ASingleBitWithSize(
      wrap.as(makeType<BShrTy>(meta.dst_ring, 1)),
      meta.dst_width - meta.src_width);
}

}  // namespace spu::mpc::cheetah
