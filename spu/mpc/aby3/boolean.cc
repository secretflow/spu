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

#include "spu/mpc/aby3/boolean.h"

#include <algorithm>

#include "yasl/utils/parallel.h"

#include "spu/core/profile.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/util/bit_utils.h"
#include "spu/mpc/util/communicator.h"

namespace spu::mpc::aby3 {

void CommonTypeB::evaluate(EvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_KERNEL(ctx, lhs, rhs);

  const size_t lhs_nbits = lhs.as<BShrTy>()->nbits();
  const size_t rhs_nbits = rhs.as<BShrTy>()->nbits();

  const size_t out_nbits = std::max(lhs_nbits, rhs_nbits);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  ctx->setOutput(makeType<BShrTy>(out_btype, out_nbits));
}

void CastTypeB::evaluate(EvalContext* ctx) const {
  const auto& in = ctx->getParam<ArrayRef>(0);
  const auto& to_type = ctx->getParam<Type>(1);

  SPU_TRACE_KERNEL(ctx, in, to_type);

  ArrayRef out(to_type, in.numel());
  DISPATCH_UINT_PT_TYPES(in.eltype().as<BShrTy>()->getBacktype(), "_", [&]() {
    using InT = ScalarT;
    DISPATCH_UINT_PT_TYPES(to_type.as<BShrTy>()->getBacktype(), "_", [&]() {
      using OutT = ScalarT;
      auto _in = ArrayView<std::array<InT, 2>>(in);
      auto _out = ArrayView<std::array<OutT, 2>>(out);

      for (auto idx = 0; idx < in.numel(); idx++) {
        _out[idx][0] = static_cast<OutT>(_in[idx][0]);
        _out[idx][1] = static_cast<OutT>(_in[idx][1]);
      }
    });
  });

  ctx->setOutput(out);
}

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();
  const PtType btype = in.eltype().as<BShrTy>()->getBacktype();
  const auto field = ctx->caller()->getState<Aby3State>()->getDefaultField();

  return DISPATCH_UINT_PT_TYPES(btype, "aby3.b2p", [&]() {
    using BShrT = ScalarT;

    return DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using PShrT = ring2k_t;

      ArrayRef out(makeType<Pub2kTy>(field), in.numel());

      auto _in = ArrayView<std::array<BShrT, 2>>(in);
      auto _out = ArrayView<PShrT>(out);

      auto x2 = getShareAs<BShrT>(in, 1);
      auto x3 = comm->rotate<BShrT>(x2, "b2p");  // comm => 1, k

      for (auto idx = 0; idx < in.numel(); idx++) {
        _out[idx] = static_cast<PShrT>(_in[idx][0] ^ _in[idx][1] ^ x3[idx]);
      }

      return out;
    });
  });
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();
  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto _in = ArrayView<ring2k_t>(in);
    const size_t nbits = maxBitWidth(_in);
    const PtType btype = calcBShareBacktype(nbits);

    return DISPATCH_UINT_PT_TYPES(btype, "_", [&]() {
      using BShrT = ScalarT;
      ArrayRef out(makeType<BShrTy>(btype, nbits), in.numel());
      auto _out = ArrayView<std::array<BShrT, 2>>(out);

      for (int64_t idx = 0; idx < in.numel(); idx++) {
        if (comm->getRank() == 0) {
          _out[idx][0] = static_cast<BShrT>(_in[idx]);
          _out[idx][1] = 0u;
        } else if (comm->getRank() == 1) {
          _out[idx][0] = 0u;
          _out[idx][1] = 0u;
        } else {
          _out[idx][0] = 0u;
          _out[idx][1] = static_cast<BShrT>(_in[idx]);
        }
      }
      return out;
    });
  });
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), "_", [&]() {
    using RhsT = ring2k_t;
    auto _rhs = ArrayView<RhsT>(rhs);
    const size_t rhs_nbits = maxBitWidth(_rhs);
    const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using LhsT = ScalarT;
      auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using OutT = ScalarT;

        ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());
        auto _out = ArrayView<std::array<OutT, 2>>(out);
        for (int64_t idx = 0; idx < lhs.numel(); idx++) {
          _out[idx][0] = _lhs[idx][0] & _rhs[idx];
          _out[idx][1] = _lhs[idx][1] & _rhs[idx];
        }
        return out;
      });
    });
  });
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto* comm = ctx->caller()->getState<Communicator>();

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using RhsT = ScalarT;
    auto _rhs = ArrayView<std::array<RhsT, 2>>(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using LhsT = ScalarT;
      auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using OutT = ScalarT;

        std::vector<OutT> r0(lhs.numel());
        std::vector<OutT> r1(lhs.numel());
        prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

        // z1 = (x1 & y1) ^ (x1 & y2) ^ (x2 & y1) ^ (r0 ^ r1);
        for (int64_t idx = 0; idx < lhs.numel(); idx++) {
          r0[idx] = (_lhs[idx][0] & _rhs[idx][0]) ^
                    (_lhs[idx][0] & _rhs[idx][1]) ^
                    (_lhs[idx][1] & _rhs[idx][0]) ^ (r0[idx] ^ r1[idx]);
        }

        r1 = comm->rotate<OutT>(r0, "andbb");  // comm => 1, k
                                               //
        auto _out = ArrayView<std::array<OutT, 2>>(out);
        for (int64_t idx = 0; idx < lhs.numel(); idx++) {
          _out[idx][0] = r0[idx];
          _out[idx][1] = r1[idx];
        }
        return out;
      });
    });
  });
}

ArrayRef XorBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), "_", [&]() {
    using RhsT = ring2k_t;
    auto _rhs = ArrayView<RhsT>(rhs);

    const size_t rhs_nbits = maxBitWidth(_rhs);
    const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);
    ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using LhsT = ScalarT;
      auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using OutT = ScalarT;

        auto _out = ArrayView<std::array<OutT, 2>>(out);
        for (int64_t idx = 0; idx < lhs.numel(); idx++) {
          _out[idx][0] = _lhs[idx][0] ^ _rhs[idx];
          _out[idx][1] = _lhs[idx][1] ^ _rhs[idx];
        }
        return out;
      });
    });
  });
}

ArrayRef XorBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using RhsT = ScalarT;
    auto _rhs = ArrayView<std::array<RhsT, 2>>(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using LhsT = ScalarT;
      auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using OutT = ScalarT;

        ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());
        auto _out = ArrayView<std::array<OutT, 2>>(out);

        for (int64_t idx = 0; idx < lhs.numel(); idx++) {
          _out[idx][0] = _lhs[idx][0] ^ _rhs[idx][0];
          _out[idx][1] = _lhs[idx][1] ^ _rhs[idx][1];
        }
        return out;
      });
    });
  });
}

ArrayRef LShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto* in_ty = in.eltype().as<BShrTy>();

  // TODO: the hal dtype should tell us about the max number of possible bits.
  const auto field = ctx->caller()->getState<Aby3State>()->getDefaultField();
  const size_t out_nbits = std::min(in_ty->nbits() + bits, SizeOf(field) * 8);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using InT = ScalarT;

    return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using OutT = ScalarT;

      ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.numel());

      auto _in = ArrayView<std::array<InT, 2>>(in);
      auto _out = ArrayView<std::array<OutT, 2>>(out);

      for (auto idx = 0; idx < in.numel(); idx++) {
        _out[idx][0] = static_cast<OutT>(_in[idx][0]) << bits;
        _out[idx][1] = static_cast<OutT>(_in[idx][1]) << bits;
      }

      return out;
    });
  });
}

ArrayRef RShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto* in_ty = in.eltype().as<BShrTy>();

  bits = std::min(in_ty->nbits(), bits);
  size_t out_nbits = in_ty->nbits();
  out_nbits -= std::min(out_nbits, bits);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using InT = ScalarT;

    return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using OutT = ScalarT;

      ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.numel());

      auto _in = ArrayView<std::array<InT, 2>>(in);
      auto _out = ArrayView<std::array<OutT, 2>>(out);

      for (auto idx = 0; idx < in.numel(); idx++) {
        _out[idx][0] = static_cast<OutT>(_in[idx][0] >> bits);
        _out[idx][1] = static_cast<OutT>(_in[idx][1] >> bits);
      }

      return out;
    });
  });
}

ArrayRef ARShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = ctx->caller()->getState<Aby3State>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();

  // arithmetic right shift expects to work on ring, or the behaviour is
  // undefined.
  YASL_ENFORCE(in_ty->nbits() == SizeOf(field) * 8, "in.type={}, field={}",
               in.eltype(), field);
  const PtType out_btype = in_ty->getBacktype();
  const size_t out_nbits = in_ty->nbits();

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using T = std::make_signed_t<ScalarT>;
    ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.numel());

    auto _in = ArrayView<std::array<T, 2>>(in);
    auto _out = ArrayView<std::array<T, 2>>(out);

    for (auto idx = 0; idx < in.numel(); idx++) {
      _out[idx][0] = _in[idx][0] >> bits;
      _out[idx][1] = _in[idx][1] >> bits;
    }

    return out;
  });
}

ArrayRef BitrevB::proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                       size_t end) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, start, end);

  YASL_ENFORCE(start <= end && end <= 128);

  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t out_nbits = std::max(in_ty->nbits(), end);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using InT = ScalarT;

    return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using OutT = ScalarT;

      ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.numel());

      auto _in = ArrayView<std::array<InT, 2>>(in);
      auto _out = ArrayView<std::array<OutT, 2>>(out);

      auto bitrev_fn = [&](OutT el) -> OutT {
        OutT tmp = 0U;
        for (size_t idx = start; idx < end; idx++) {
          if (el & ((OutT)1 << idx)) {
            tmp |= (OutT)1 << (end - 1 - idx + start);
          }
        }

        OutT mask = ((OutT)1U << end) - ((OutT)1U << start);
        return (el & ~mask) | tmp;
      };

      for (auto idx = 0; idx < in.numel(); idx++) {
        _out[idx][0] = bitrev_fn(static_cast<OutT>(_in[idx][0]));
        _out[idx][1] = bitrev_fn(static_cast<OutT>(_in[idx][1]));
      }

      return out;
    });
  });
}

}  // namespace spu::mpc::aby3
