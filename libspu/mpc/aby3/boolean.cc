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

#include "libspu/mpc/aby3/boolean.h"

#include <algorithm>

#include "yacl/utils/platform_utils.h"

#include "libspu/core/bit_utils.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc::aby3 {

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  const size_t lhs_nbits = lhs.as<BShrTy>()->nbits();
  const size_t rhs_nbits = rhs.as<BShrTy>()->nbits();

  const size_t out_nbits = std::max(lhs_nbits, rhs_nbits);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  ctx->setOutput(makeType<BShrTy>(out_btype, out_nbits));
}

NdArrayRef CastTypeB::proc(KernelEvalContext*, const NdArrayRef& in,
                           const Type& to_type) const {
  NdArrayRef out(to_type, in.shape());
  DISPATCH_UINT_PT_TYPES(in.eltype().as<BShrTy>()->getBacktype(), "_", [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;

    DISPATCH_UINT_PT_TYPES(to_type.as<BShrTy>()->getBacktype(), "_", [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayView<out_shr_t> _out(out);
      NdArrayView<in_shr_t> _in(in);

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx][0] = static_cast<out_el_t>(v[0]);
        _out[idx][1] = static_cast<out_el_t>(v[1]);
      });
    });
  });

  return out;
}

NdArrayRef B2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const PtType btype = in.eltype().as<BShrTy>()->getBacktype();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  return DISPATCH_UINT_PT_TYPES(btype, "aby3.b2p", [&]() {
    using bshr_el_t = ScalarT;
    using bshr_t = std::array<bshr_el_t, 2>;

    return DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using pshr_el_t = ring2k_t;

      NdArrayRef out(makeType<Pub2kTy>(field), in.shape());

      NdArrayView<pshr_el_t> _out(out);
      NdArrayView<bshr_t> _in(in);

      auto x2 = getShareAs<bshr_el_t>(in, 1);
      auto x3 = comm->rotate<bshr_el_t>(x2, "b2p");  // comm => 1, k

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx] = static_cast<pshr_el_t>(v[0] ^ v[1] ^ x3[idx]);
      });

      return out;
    });
  });
}

NdArrayRef P2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    const size_t nbits = maxBitWidth<ring2k_t>(in);
    const PtType btype = calcBShareBacktype(nbits);
    NdArrayView<ring2k_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(btype, "_", [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 2>;

      NdArrayRef out(makeType<BShrTy>(btype, nbits), in.shape());
      NdArrayView<bshr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        if (comm->getRank() == 0) {
          _out[idx][0] = static_cast<bshr_el_t>(_in[idx]);
          _out[idx][1] = 0U;
        } else if (comm->getRank() == 1) {
          _out[idx][0] = 0U;
          _out[idx][1] = 0U;
        } else {
          _out[idx][0] = 0U;
          _out[idx][1] = static_cast<bshr_el_t>(_in[idx]);
        }
      });
      return out;
    });
  });
}

NdArrayRef B2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const PtType btype = in.eltype().as<BShrTy>()->getBacktype();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  return DISPATCH_UINT_PT_TYPES(btype, "aby3.b2v", [&]() {
    using bshr_el_t = ScalarT;
    using bshr_t = std::array<bshr_el_t, 2>;
    NdArrayView<bshr_t> _in(in);

    return DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using vshr_scalar_t = ring2k_t;

      auto out_ty = makeType<Priv2kTy>(field, rank);

      if (comm->getRank() == rank) {
        auto x3 =
            comm->recv<bshr_el_t>(comm->nextRank(), "b2v");  // comm => 1, k

        NdArrayRef out(out_ty, in.shape());
        NdArrayView<vshr_scalar_t> _out(out);

        pforeach(0, in.numel(), [&](int64_t idx) {
          const auto& v = _in[idx];
          _out[idx] = v[0] ^ v[1] ^ x3[idx];
        });
        return out;
      } else if (comm->getRank() == (rank + 1) % 3) {
        std::vector<bshr_el_t> x2(in.numel());

        pforeach(0, in.numel(), [&](int64_t idx) { x2[idx] = _in[idx][1]; });

        comm->sendAsync<bshr_el_t>(comm->prevRank(), x2,
                                   "b2v");  // comm => 1, k

        return makeConstantArrayRef(out_ty, in.shape());
      } else {
        return makeConstantArrayRef(out_ty, in.shape());
      }
    });
  });
}

NdArrayRef AndBP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), "_", [&]() {
    using rhs_scalar_t = ring2k_t;

    const size_t rhs_nbits = maxBitWidth<rhs_scalar_t>(rhs);
    const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);

    NdArrayView<rhs_scalar_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());
        NdArrayView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] & r;
          _out[idx][1] = l[1] & r;
        });

        return out;
      });
    });
  });
}

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        std::vector<out_el_t> r0(lhs.numel());
        std::vector<out_el_t> r1(lhs.numel());
        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);

        // z1 = (x1 & y1) ^ (x1 & y2) ^ (x2 & y1) ^ (r0 ^ r1);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          r0[idx] = (l[0] & r[0]) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^
                    (r0[idx] ^ r1[idx]);
        });

        r1 = comm->rotate<out_el_t>(r0, "andbb");  // comm => 1, k

        NdArrayView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out[idx][0] = r0[idx];
          _out[idx][1] = r1[idx];
        });
        return out;
      });
    });
  });
}

NdArrayRef XorBP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), "_", [&]() {
    using rhs_scalar_t = ring2k_t;

    const size_t rhs_nbits = maxBitWidth<rhs_scalar_t>(rhs);
    const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);

    NdArrayView<rhs_scalar_t> _rhs(rhs);

    NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        NdArrayView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] ^ r;
          _out[idx][1] = l[1] ^ r;
        });
        return out;
      });
    });
  });
}

NdArrayRef XorBB::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;

    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());
        NdArrayView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] ^ r[0];
          _out[idx][1] = l[1] ^ r[1];
        });
        return out;
      });
    });
  });
}

NdArrayRef LShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         size_t bits) const {
  const auto* in_ty = in.eltype().as<BShrTy>();

  // TODO: the hal dtype should tell us about the max number of possible bits.
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::min(in_ty->nbits() + bits, SizeOf(field) * 8);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;

    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
      NdArrayView<out_shr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx][0] = static_cast<out_el_t>(v[0]) << bits;
        _out[idx][1] = static_cast<out_el_t>(v[1]) << bits;
      });

      return out;
    });
  });
}

NdArrayRef RShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t bits) const {
  const auto* in_ty = in.eltype().as<BShrTy>();

  bits = std::min(in_ty->nbits(), bits);
  size_t out_nbits = in_ty->nbits();
  out_nbits -= std::min(out_nbits, bits);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using in_shr_t = std::array<ScalarT, 2>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
      NdArrayView<out_shr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx][0] = static_cast<out_el_t>(v[0] >> bits);
        _out[idx][1] = static_cast<out_el_t>(v[1] >> bits);
      });

      return out;
    });
  });
}

NdArrayRef ARShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();

  // arithmetic right shift expects to work on ring, or the behaviour is
  // undefined.
  SPU_ENFORCE(in_ty->nbits() == SizeOf(field) * 8, "in.type={}, field={}",
              in.eltype(), field);
  const PtType out_btype = in_ty->getBacktype();
  const size_t out_nbits = in_ty->nbits();

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using el_t = std::make_signed_t<ScalarT>;
    using shr_t = std::array<el_t, 2>;

    NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      const auto& v = _in[idx];
      _out[idx][0] = v[0] >> bits;
      _out[idx][1] = v[1] >> bits;
    });

    return out;
  });
}

NdArrayRef BitrevB::proc(KernelEvalContext*, const NdArrayRef& in, size_t start,
                         size_t end) const {
  SPU_ENFORCE(start <= end && end <= 128);

  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t out_nbits = std::max(in_ty->nbits(), end);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;

    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
      NdArrayView<out_shr_t> _out(out);

      auto bitrev_fn = [&](out_el_t el) -> out_el_t {
        out_el_t tmp = 0U;
        for (size_t idx = start; idx < end; idx++) {
          if (el & ((out_el_t)1 << idx)) {
            tmp |= (out_el_t)1 << (end - 1 - idx + start);
          }
        }

        out_el_t mask = ((out_el_t)1U << end) - ((out_el_t)1U << start);
        return (el & ~mask) | tmp;
      };

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx][0] = bitrev_fn(static_cast<out_el_t>(v[0]));
        _out[idx][1] = bitrev_fn(static_cast<out_el_t>(v[1]));
      });

      return out;
    });
  });
}

NdArrayRef BitIntlB::proc(KernelEvalContext*, const NdArrayRef& in,
                          size_t stride) const {
  // void BitIntlB::evaluate(KernelEvalContext* ctx) const {
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 2>;
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      const auto& v = _in[idx];
      _out[idx][0] = BitIntl<el_t>(v[0], stride, nbits);
      _out[idx][1] = BitIntl<el_t>(v[1], stride, nbits);
    });
  });

  return out;
}

NdArrayRef BitDeintlB::proc(KernelEvalContext*, const NdArrayRef& in,
                            size_t stride) const {
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 2>;
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      const auto& v = _in[idx];
      _out[idx][0] = BitDeintl<el_t>(v[0], stride, nbits);
      _out[idx][1] = BitDeintl<el_t>(v[1], stride, nbits);
    });
  });

  return out;
}

}  // namespace spu::mpc::aby3
