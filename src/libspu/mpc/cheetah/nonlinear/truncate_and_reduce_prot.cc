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

#include "libspu/mpc/cheetah/nonlinear/truncate_and_reduce_prot.h"

#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

namespace {

// hack for invalid NdArray
bool isValidNdArray(const NdArrayRef& x) { return x.elsize() != 0; }

template <typename T>
typename std::make_unsigned<T>::type makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

}  // namespace

NdArrayRef RingTruncateAndReduceProtocol::Compute(const NdArrayRef& inp,
                                                  const NdArrayRef& wrap_s,
                                                  const Meta& meta) {
  auto src_field = inp.eltype().as<RingTy>()->field();
  // sanity check
  {
    SPU_ENFORCE(src_field == meta.src_ring);
    SPU_ENFORCE(meta.src_width >= meta.dst_width);
    SPU_ENFORCE(meta.dst_width <= (int64_t)SizeOf(meta.dst_ring) * 8);
  }

  // same ring
  if (meta.src_width == meta.dst_width) {
    return inp;
  }

  if (!meta.exact) {
    return ComputeApprox(inp, meta);
  }

  if (isValidNdArray(wrap_s)) {
    return ComputeWithWrap(inp, wrap_s, meta);
  }

  return ComputeWithoutWrap(inp, meta);
}

NdArrayRef RingTruncateAndReduceProtocol::ComputeWithoutWrap(
    const NdArrayRef& inp, const Meta& meta) {
  // let tr_bits = k
  // w = 1{v0 + v1 > 2^k - 1}
  //   = 1{v0 > 2^k - 1 - v1}
  const auto tr_bits = meta.src_width - meta.dst_width;
  auto v = ring_reduce(inp, tr_bits);

  const int rank = basic_ot_prot_->Rank();
  NdArrayRef wrap_s;
  CompareProtocol compare_prot(basic_ot_prot_);
  if (rank == 0) {
    wrap_s = compare_prot.Compute(v, /*gt*/ true, tr_bits);
  } else {
    DISPATCH_ALL_FIELDS(meta.src_ring, [&]() {
      using st = ring2k_t;
      // 2^k - 1
      const auto mask = makeMask<st>(tr_bits);
      NdArrayView<st> v_(v);
      pforeach(0, inp.numel(),
               [&](int64_t idx) { v_[idx] = (mask - v_[idx]) & mask; });
    });
    wrap_s = compare_prot.Compute(v, /*gt*/ true, tr_bits);
  }

  return ComputeWithWrap(inp, wrap_s.as(makeType<BShrTy>(meta.src_ring, 1)),
                         meta);
}

NdArrayRef RingTruncateAndReduceProtocol::ComputeWithWrap(
    const NdArrayRef& inp, const NdArrayRef& wrap_s, const Meta& meta) {
  SPU_ENFORCE(meta.exact && isValidNdArray(wrap_s));
  SPU_ENFORCE(inp.shape() == wrap_s.shape());
  SPU_ENFORCE(wrap_s.eltype().isa<BShrTy>(), "wrap must be bshare");
  SPU_ENFORCE(wrap_s.eltype().as<BShrTy>()->nbits() == 1,
              "Supported only for 1 bit.");

  const auto tr_bits = meta.src_width - meta.dst_width;

  auto wrap_s_ashr =
      basic_ot_prot_->B2ASingleBitWithSize(wrap_s, meta.dst_width);

  NdArrayRef out = ring_zeros(meta.dst_ring, inp.shape());
  out.set_fxp_bits(meta.dst_width);

  DISPATCH_ALL_FIELDS(meta.src_ring, [&]() {
    using ust = typename std::make_unsigned<ring2k_t>::type;

    // view as unsigned type, for logical right shift to access u0 and u1
    NdArrayView<const ust> u_inp_(inp);

    DISPATCH_ALL_FIELDS(wrap_s_ashr.eltype().as<RingTy>()->field(), [&]() {
      using wrap_t = typename std::make_unsigned<ring2k_t>::type;
      NdArrayView<wrap_t> u_wrap_s_ashr_(wrap_s_ashr);

      DISPATCH_ALL_FIELDS(meta.dst_ring, [&]() {
        using udt = typename std::make_unsigned<ring2k_t>::type;
        NdArrayView<udt> u_out_(out);

        auto mask = makeMask<udt>(meta.dst_width);

        pforeach(0, inp.numel(), [&](int64_t idx) {
          auto u = static_cast<udt>(u_inp_[idx] >> tr_bits);
          u_out_[idx] = (u + u_wrap_s_ashr_[idx]) & mask;
        });
      });
    });
  });

  return out;
}

// Purely free protocol
// just ignore the wrap of truncated part, then:
// error = gt - out = 0 or 1
NdArrayRef RingTruncateAndReduceProtocol::ComputeApprox(const NdArrayRef& inp,
                                                        const Meta& meta) {
  // let x = u || v, x0 = u0 || v0, x1 = u1 || v1
  // y = (x>>s) mod 2^{k-s}
  // then y = u0 + u1 mod 2^{k-s}
  SPU_ENFORCE(meta.exact == false);

  const auto tr_bits = meta.src_width - meta.dst_width;
  NdArrayRef out = ring_zeros(meta.dst_ring, inp.shape());
  out.set_fxp_bits(meta.dst_width);

  DISPATCH_ALL_FIELDS(meta.src_ring, [&]() {
    using ust = typename std::make_unsigned<ring2k_t>::type;

    // view as unsigned type, for logical right shift to access u0 and u1
    NdArrayView<const ust> u_inp_(inp);

    DISPATCH_ALL_FIELDS(meta.dst_ring, [&]() {
      using udt = typename std::make_unsigned<ring2k_t>::type;
      NdArrayView<udt> u_out_(out);
      auto mask = makeMask<udt>(meta.dst_width);

      pforeach(0, inp.numel(), [&](int64_t i) {
        auto u = u_inp_[i] >> tr_bits;
        u_out_[i] = static_cast<udt>(u) & mask;
      });
    });
  });

  return out;
}

}  // namespace spu::mpc::cheetah