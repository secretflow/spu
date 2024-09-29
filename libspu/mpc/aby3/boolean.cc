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

#include "libspu/core/bit_utils.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/utils.h"

namespace spu::mpc::aby3 {

size_t getNumBits(const MemRef& in) {
  if (in.eltype().isa<Pub2kTy>()) {
    return DISPATCH_ALL_STORAGE_TYPES(
        in.eltype().storage_type(), [&]() { return maxBitWidth<ScalarT>(in); });
  } else if (in.eltype().isa<BoolShareTy>()) {
    return in.eltype().as<BoolShareTy>()->valid_bits();
  } else {
    SPU_THROW("should not be here, {}", in.eltype());
  }
}

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  const size_t out_nbits = std::max(lhs.as<BoolShareTy>()->valid_bits(),
                                    rhs.as<BoolShareTy>()->valid_bits());

  ctx->pushOutput(
      makeType<BoolShareTy>(std::max(lhs.semantic_type(), rhs.semantic_type()),
                            GetStorageType(out_nbits), out_nbits));
}

MemRef CastTypeB::proc(KernelEvalContext*, const MemRef& in,
                       const Type& to_type) const {
  MemRef out(to_type, in.shape());
  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;

    MemRefView<in_shr_t> _in(in);

    DISPATCH_ALL_STORAGE_TYPES(to_type.as<BoolShareTy>()->storage_type(),
                               [&]() {
                                 using out_el_t = ScalarT;
                                 using out_shr_t = std::array<out_el_t, 2>;

                                 MemRefView<out_shr_t> _out(out);

                                 pforeach(0, in.numel(), [&](int64_t idx) {
                                   const auto& v = _in[idx];
                                   _out[idx][0] = static_cast<out_el_t>(v[0]);
                                   _out[idx][1] = static_cast<out_el_t>(v[1]);
                                 });
                               });
  });

  return out;
}

MemRef B2P::proc(KernelEvalContext* ctx, const MemRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  auto out_ty = makeType<Pub2kTy>(in.eltype().semantic_type());

  return DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using bshr_el_t = ScalarT;
    using bshr_t = std::array<bshr_el_t, 2>;

    MemRefView<bshr_t> _in(in);
    auto x2 = getShareAs<bshr_el_t>(in, 1);
    auto x3 = comm->rotate<bshr_el_t>(x2, "b2p");  // comm => 1, k

    return DISPATCH_ALL_STORAGE_TYPES(out_ty.storage_type(), [&]() {
      using pshr_el_t = ScalarT;

      MemRef out(out_ty, in.shape());
      MemRefView<pshr_el_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx] = static_cast<pshr_el_t>(v[0] ^ v[1] ^ x3[idx]);
      });

      return out;
    });
  });
}

MemRef P2B::proc(KernelEvalContext* ctx, const MemRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  auto field = getNumBits(in);
  MemRef out(makeType<BoolShareTy>(in.eltype().semantic_type(),
                                   GetStorageType(field), field),
             in.shape());

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _in(in);

    DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 2>;

      MemRefView<bshr_t> _out(out);

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
    });
  });

  return out;
}

MemRef B2V::proc(KernelEvalContext* ctx, const MemRef& in, size_t rank) const {
  auto* comm = ctx->getState<Communicator>();

  auto out_ty = makeType<Priv2kTy>(in.eltype().semantic_type(), rank);

  return DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using bshr_el_t = ScalarT;
    using bshr_t = std::array<bshr_el_t, 2>;
    MemRefView<bshr_t> _in(in);

    return DISPATCH_ALL_STORAGE_TYPES(out_ty.storage_type(), [&]() {
      using vshr_scalar_t = ScalarT;

      if (comm->getRank() == rank) {
        auto x3 =
            comm->recv<bshr_el_t>(comm->nextRank(), "b2v");  // comm => 1, k

        MemRef out(out_ty, in.shape());
        MemRefView<vshr_scalar_t> _out(out);

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

MemRef AndBP::proc(KernelEvalContext*, const MemRef& lhs,
                   const MemRef& rhs) const {
  size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  MemRef out(makeType<BoolShareTy>(lhs.eltype().semantic_type(),
                                   GetStorageType(out_nbits), out_nbits),
             lhs.shape());

  DISPATCH_ALL_STORAGE_TYPES(rhs.eltype().storage_type(), [&]() {
    using rhs_scalar_t = ScalarT;
    MemRefView<rhs_scalar_t> _rhs(rhs);

    DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      MemRefView<lhs_shr_t> _lhs(lhs);

      DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        MemRefView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] & r;
          _out[idx][1] = l[1] & r;
        });
      });
    });
  });

  return out;
}

MemRef AndBB::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  MemRef out(makeType<BoolShareTy>(lhs.eltype().semantic_type(),
                                   GetStorageType(out_nbits), out_nbits),
             lhs.shape());

  DISPATCH_ALL_STORAGE_TYPES(rhs.eltype().storage_type(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    MemRefView<rhs_shr_t> _rhs(rhs);

    DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      MemRefView<lhs_shr_t> _lhs(lhs);

      DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        std::vector<out_el_t> r0(lhs.numel());
        std::vector<out_el_t> r1(lhs.numel());
        prg_state->fillPrssPair(r0.data(), r1.data(), GetVectorNumBytes(r0));

        // z1 = (x1 & y1) ^ (x1 & y2) ^ (x2 & y1) ^ (r0 ^ r1);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          r0[idx] = (l[0] & r[0]) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^
                    (r0[idx] ^ r1[idx]);
        });

        r1 = comm->rotate<out_el_t>(r0, "andbb");  // comm => 1, k

        MemRefView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out[idx][0] = r0[idx];
          _out[idx][1] = r1[idx];
        });
      });
    });
  });

  return out;
}

MemRef XorBP::proc(KernelEvalContext*, const MemRef& lhs,
                   const MemRef& rhs) const {
  size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));
  MemRef out(makeType<BoolShareTy>(lhs.eltype().semantic_type(),
                                   GetStorageType(out_nbits), out_nbits),
             lhs.shape());

  DISPATCH_ALL_STORAGE_TYPES(rhs.eltype().storage_type(), [&]() {
    using rhs_scalar_t = ScalarT;
    MemRefView<rhs_scalar_t> _rhs(rhs);

    DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      MemRefView<lhs_shr_t> _lhs(lhs);

      DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        MemRefView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] ^ r;
          _out[idx][1] = l[1] ^ r;
        });
      });
    });
  });

  return out;
}

MemRef XorBB::proc(KernelEvalContext*, const MemRef& lhs,
                   const MemRef& rhs) const {
  size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));
  MemRef out(makeType<BoolShareTy>(lhs.eltype().semantic_type(),
                                   GetStorageType(out_nbits), out_nbits),
             lhs.shape());

  DISPATCH_ALL_STORAGE_TYPES(rhs.eltype().storage_type(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;

    MemRefView<rhs_shr_t> _rhs(rhs);

    DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      MemRefView<lhs_shr_t> _lhs(lhs);

      DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        MemRefView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] ^ r[0];
          _out[idx][1] = l[1] ^ r[1];
        });
      });
    });
  });

  return out;
}

}  // namespace spu::mpc::aby3
