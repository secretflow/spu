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
#include "libspu/core/platform_utils.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"

namespace spu::mpc::aby3 {

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const size_t lhs_nbits = lhs.as<BShrTy>()->nbits();
  const size_t rhs_nbits = rhs.as<BShrTy>()->nbits();

  const size_t out_nbits = std::max(lhs_nbits, rhs_nbits);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  ctx->setOutput(makeType<BShrTy>(out_btype, out_nbits));
}

void CastTypeB::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<ArrayRef>(0);
  const auto& to_type = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_LEAF(ctx, in, to_type);

  ArrayRef out(to_type, in.numel());
  DISPATCH_UINT_PT_TYPES(in.eltype().as<BShrTy>()->getBacktype(), "_", [&]() {
    using InT = ScalarT;
    DISPATCH_UINT_PT_TYPES(to_type.as<BShrTy>()->getBacktype(), "_", [&]() {
      using OutT = ScalarT;
      auto _in = ArrayView<std::array<InT, 2>>(in);
      auto _out = ArrayView<std::array<OutT, 2>>(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = static_cast<OutT>(_in[idx][0]);
        _out[idx][1] = static_cast<OutT>(_in[idx][1]);
      });
    });
  });

  ctx->setOutput(out);
}

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const PtType btype = in.eltype().as<BShrTy>()->getBacktype();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  return DISPATCH_UINT_PT_TYPES(btype, "aby3.b2p", [&]() {
    using BShrT = ScalarT;

    return DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using PShrT = ring2k_t;

      ArrayRef out(makeType<Pub2kTy>(field), in.numel());

      auto _in = ArrayView<std::array<BShrT, 2>>(in);
      auto _out = ArrayView<PShrT>(out);

      auto x2 = getShareAs<BShrT>(in, 1);
      auto x3 = comm->rotate<BShrT>(x2, "b2p");  // comm => 1, k

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = static_cast<PShrT>(_in[idx][0] ^ _in[idx][1] ^ x3[idx]);
      });

      return out;
    });
  });
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto _in = ArrayView<ring2k_t>(in);
    const size_t nbits = _in.maxBitWidth();
    const PtType btype = calcBShareBacktype(nbits);

    return DISPATCH_UINT_PT_TYPES(btype, "_", [&]() {
      using BShrT = ScalarT;
      ArrayRef out(makeType<BShrTy>(btype, nbits), in.numel());
      auto _out = ArrayView<std::array<BShrT, 2>>(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        if (comm->getRank() == 0) {
          _out[idx][0] = static_cast<BShrT>(_in[idx]);
          _out[idx][1] = 0U;
        } else if (comm->getRank() == 1) {
          _out[idx][0] = 0U;
          _out[idx][1] = 0U;
        } else {
          _out[idx][0] = 0U;
          _out[idx][1] = static_cast<BShrT>(_in[idx]);
        }
      });
      return out;
    });
  });
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), "_", [&]() {
    using RhsT = ring2k_t;
    auto _rhs = ArrayView<RhsT>(rhs);
    const size_t rhs_nbits = _rhs.maxBitWidth();
    const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using LhsT = ScalarT;
      auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using OutT = ScalarT;

        ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());
        auto _out = ArrayView<std::array<OutT, 2>>(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out[idx][0] = _lhs[idx][0] & _rhs[idx];
          _out[idx][1] = _lhs[idx][1] & _rhs[idx];
        });

        return out;
      });
    });
  });
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

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
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          r0[idx] = (_lhs[idx][0] & _rhs[idx][0]) ^
                    (_lhs[idx][0] & _rhs[idx][1]) ^
                    (_lhs[idx][1] & _rhs[idx][0]) ^ (r0[idx] ^ r1[idx]);
        });

        r1 = comm->rotate<OutT>(r0, "andbb");  // comm => 1, k

        auto _out = ArrayView<std::array<OutT, 2>>(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out[idx][0] = r0[idx];
          _out[idx][1] = r1[idx];
        });
        return out;
      });
    });
  });
}

ArrayRef XorBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), "_", [&]() {
    using RhsT = ring2k_t;
    auto _rhs = ArrayView<RhsT>(rhs);

    const size_t rhs_nbits = _rhs.maxBitWidth();
    const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);
    ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using LhsT = ScalarT;
      auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using OutT = ScalarT;

        auto _out = ArrayView<std::array<OutT, 2>>(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out[idx][0] = _lhs[idx][0] ^ _rhs[idx];
          _out[idx][1] = _lhs[idx][1] ^ _rhs[idx];
        });
        return out;
      });
    });
  });
}

ArrayRef XorBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

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

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out[idx][0] = _lhs[idx][0] ^ _rhs[idx][0];
          _out[idx][1] = _lhs[idx][1] ^ _rhs[idx][1];
        });
        return out;
      });
    });
  });
}

ArrayRef LShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto* in_ty = in.eltype().as<BShrTy>();

  // TODO: the hal dtype should tell us about the max number of possible bits.
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::min(in_ty->nbits() + bits, SizeOf(field) * 8);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using InT = ScalarT;

    return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using OutT = ScalarT;

      ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.numel());

      auto _in = ArrayView<std::array<InT, 2>>(in);
      auto _out = ArrayView<std::array<OutT, 2>>(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = static_cast<OutT>(_in[idx][0]) << bits;
        _out[idx][1] = static_cast<OutT>(_in[idx][1]) << bits;
      });

      return out;
    });
  });
}

ArrayRef RShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

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

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = static_cast<OutT>(_in[idx][0] >> bits);
        _out[idx][1] = static_cast<OutT>(_in[idx][1] >> bits);
      });

      return out;
    });
  });
}

ArrayRef ARShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();

  // arithmetic right shift expects to work on ring, or the behaviour is
  // undefined.
  SPU_ENFORCE(in_ty->nbits() == SizeOf(field) * 8, "in.type={}, field={}",
              in.eltype(), field);
  const PtType out_btype = in_ty->getBacktype();
  const size_t out_nbits = in_ty->nbits();

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using T = std::make_signed_t<ScalarT>;
    ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.numel());

    auto _in = ArrayView<std::array<T, 2>>(in);
    auto _out = ArrayView<std::array<T, 2>>(out);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = _in[idx][0] >> bits;
      _out[idx][1] = _in[idx][1] >> bits;
    });

    return out;
  });
}

ArrayRef BitrevB::proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                       size_t end) const {
  SPU_TRACE_MPC_LEAF(ctx, in, start, end);

  SPU_ENFORCE(start <= end && end <= 128);

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

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = bitrev_fn(static_cast<OutT>(_in[idx][0]));
        _out[idx][1] = bitrev_fn(static_cast<OutT>(_in[idx][1]));
      });

      return out;
    });
  });
}

namespace {

constexpr std::array<uint128_t, 6> kBitIntlSwapMasks = {{
    yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
    yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
    yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
    yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
    yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
    yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
}};

constexpr std::array<uint128_t, 6> kBitIntlKeepMasks = {{
    yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
    yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
    yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
    yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
    yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
    yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
}};

}  // namespace

void BitIntlB::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<ArrayRef>(0);
  const size_t stride = ctx->getParam<size_t>(1);

  SPU_TRACE_MPC_LEAF(ctx, in, stride);

  // algorithm:
  //      0000000011111111
  // swap     ^^^^^^^^
  //      0000111100001111
  // swap   ^^^^    ^^^^
  //      0011001100110011
  // swap  ^^  ^^  ^^  ^^
  //      0101010101010101
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  ArrayRef out = in.clone();
  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using T = ScalarT;
    auto _in = ArrayView<std::array<T, 2>>(in);
    auto _out = ArrayView<std::array<T, 2>>(out);

    if constexpr (std::is_same_v<T, uint64_t>) {
      pforeach(0, in.numel(), [&](int64_t idx) {
        constexpr std::array<uint64_t, 6> kMasks = {{
            0x5555555555555555,  // 01010101
            0x3333333333333333,  // 00110011
            0x0F0F0F0F0F0F0F0F,  // 00001111
            0x00FF00FF00FF00FF,  // ...
            0x0000FFFF0000FFFF,  // ...
            0x00000000FFFFFFFF,  // ...
        }};
        const uint64_t r0 = _in[idx][0];
        const uint64_t r1 = _in[idx][1];

        const uint64_t m = kMasks[stride];
        _out[idx][0] = pdep_u64(r0, m) ^ pdep_u64(r0 >> 32, ~m);
        _out[idx][1] = pdep_u64(r1, m) ^ pdep_u64(r1 >> 32, ~m);
      });
    } else {
      pforeach(0, in.numel(), [&](int64_t idx) {
        T r0 = _in[idx][0];
        T r1 = _in[idx][1];
        for (int64_t level = Log2Ceil(nbits) - 2;
             level >= static_cast<int64_t>(stride); level--) {
          T K = static_cast<T>(kBitIntlKeepMasks[level]);
          T M = static_cast<T>(kBitIntlSwapMasks[level]);
          int S = 1 << level;

          r0 = (r0 & K) ^ ((r0 >> S) & M) ^ ((r0 & M) << S);
          r1 = (r1 & K) ^ ((r1 >> S) & M) ^ ((r1 & M) << S);
        }
        _out[idx][0] = r0;
        _out[idx][1] = r1;
      });
    }
  });

  ctx->setOutput(out);
}

void BitDeintlB::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<ArrayRef>(0);
  const size_t stride = ctx->getParam<size_t>(1);

  SPU_TRACE_MPC_LEAF(ctx, in, stride);

  // algorithm:
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  ArrayRef out = in.clone();
  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using T = ScalarT;
    auto _in = ArrayView<std::array<T, 2>>(in);
    auto _out = ArrayView<std::array<T, 2>>(out);

    if constexpr (std::is_same_v<T, uint64_t>) {
      pforeach(0, in.numel(), [&](int64_t idx) {
        constexpr std::array<uint64_t, 6> kMasks = {{
            0x5555555555555555,  // 01010101
            0x3333333333333333,  // 00110011
            0x0F0F0F0F0F0F0F0F,  // 00001111
            0x00FF00FF00FF00FF,  // ...
            0x0000FFFF0000FFFF,  // ...
            0x00000000FFFFFFFF,  // ...
        }};
        const uint64_t r0 = _in[idx][0];
        const uint64_t r1 = _in[idx][1];

        const uint64_t m = kMasks[stride];
        _out[idx][0] = pext_u64(r0, m) ^ (pext_u64(r0, ~m) << 32);
        _out[idx][1] = pext_u64(r1, m) ^ (pext_u64(r1, ~m) << 32);
      });
    } else {
      pforeach(0, in.numel(), [&](int64_t idx) {
        T r0 = _in[idx][0];
        T r1 = _in[idx][1];
        for (int64_t level = stride; level + 1 < Log2Ceil(nbits); level++) {
          T K = static_cast<T>(kBitIntlKeepMasks[level]);
          T M = static_cast<T>(kBitIntlSwapMasks[level]);
          int S = 1 << level;

          r0 = (r0 & K) ^ ((r0 >> S) & M) ^ ((r0 & M) << S);
          r1 = (r1 & K) ^ ((r1 >> S) & M) ^ ((r1 & M) << S);
        }
        _out[idx][0] = r0;
        _out[idx][1] = r1;
      });
    }
  });

  ctx->setOutput(out);
}

}  // namespace spu::mpc::aby3
