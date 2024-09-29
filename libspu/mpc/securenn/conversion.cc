// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/securenn/conversion.h"

#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/securenn/arithmetic.h"
#include "libspu/mpc/securenn/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::securenn {

MemRef A2B::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const auto* ty = x.eltype().as<BaseRingType>();
  const auto valid_bits = ty->valid_bits();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<MemRef> bshrs;
  const auto bty = makeType<BoolShareTy>(ty->semantic_type(),
                                         ty->storage_type(), valid_bits);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    MemRef r0(x.eltype(), x.shape());
    MemRef r1(x.eltype(), x.shape());
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());
    auto b = ring_xor(r0, r1).as(bty);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }
    bshrs.push_back(b.as(bty));
  }

  MemRef res = vreduce(bshrs.begin(), bshrs.end(),
                       [&](const MemRef& xx, const MemRef& yy) {
                         return add_bb(ctx->sctx(), xx, yy);
                       });
  return res.as(bty);
}

MemRef B2A::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const auto* ty = x.eltype().as<BaseRingType>();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  const int64_t field = SizeOf(ty->storage_type()) * 8;

  MemRef r_v(
      makeType<RingTy>(ty->semantic_type(), SizeOf(ty->storage_type()) * 8),
      x.shape());
  prg_state->fillPriv(r_v.data(), r_v.elsize() * r_v.numel());
  auto r_a = r_v.as(makeType<ArithShareTy>(ty->semantic_type(),
                                           SizeOf(ty->storage_type()) * 8));

  // convert r to boolean share.
  auto r_b = a2b(ctx->sctx(), r_a);

  // evaluate adder circuit on x & r, and reveal x+r
  auto x_plus_r =
      comm->allReduce(ReduceOp::XOR, add_bb(ctx->sctx(), x, r_b), kBindName());

  // compute -r + (x+r)
  ring_neg_(r_a);
  if (comm->getRank() == 0) {
    ring_add_(r_a, x_plus_r);
  }
  if (comm->getRank() == 2) {
    comm->sendAsync(0, r_a, "r_a");
    MemRef r_a(makeType<ArithShareTy>(ty->semantic_type(), field), x.shape());
    ring_zeros(r_a);
  }
  if (comm->getRank() == 0) {
    auto tmp = comm->recv(2, makeType<ArithShareTy>(ty->semantic_type(), field),
                          "r_a");
    tmp = tmp.reshape(x.shape());
    r_a = ring_add(r_a, tmp);
  }
  return r_a;
}

MemRef B2A_Randbit::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const auto* ty = x.eltype().as<BoolShareTy>();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto rank = comm->getRank();

  const int64_t nbits = ty->valid_bits();
  const int64_t field = SizeOf(ty->storage_type()) * 8;
  if (nbits == 0) {
    // special case, it's known to be zero.
    MemRef ret(makeType<ArithShareTy>(ty->semantic_type(), field), x.shape());
    ring_zeros(ret);
    return ret;
  }

  auto numel = x.numel();
  const auto aty = makeType<ArithShareTy>(ty->semantic_type(), field);

  MemRef randbits(makeType<RingTy>(ty->semantic_type(), field),
                  {numel * nbits});
  prg_state->fillPriv(randbits.data(), randbits.elsize() * randbits.numel());

  // reconstruct ranbits
  if (rank == 0) {
    comm->sendAsync(2, randbits, "randbits0");
  }
  if (rank == 1) {
    comm->sendAsync(2, randbits, "randbits1");
  }
  if (rank == 2) {
    auto randbits0 = comm->recv(0, aty, "randbits0");
    randbits0 = randbits0.reshape(randbits.shape());
    auto randbits1 = comm->recv(1, aty, "randbits1");
    randbits1 = randbits1.reshape(randbits.shape());
    auto randbits_recon = ring_add(ring_add(randbits, randbits0), randbits1);

    MemRef randbits_(makeType<RingTy>(ty->semantic_type(), field),
                     randbits.shape());
    ring_randbit(randbits_);
    auto adjust = ring_sub(randbits_, randbits_recon);
    comm->sendAsync(0, adjust, "adjust");
  }
  if (rank == 0) {
    auto adjust = comm->recv(2, aty, "adjust");
    adjust = adjust.reshape(randbits.shape());
    ring_add_(randbits, adjust);
  }

  auto res = MemRef(aty, x.shape());

  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using U = ScalarT;

    MemRefView<U> _randbits(randbits);
    MemRefView<U> _x(x);

    // algorithm begins.
    // Ref: III.D @ https://eprint.iacr.org/2019/599.pdf (SPDZ-2K primitives)
    std::vector<U> x_xor_r(numel);

    pforeach(0, numel, [&](int64_t idx) {
      // use _r[i*nbits, (i+1)*nbits) to construct rb[i]
      U mask = 0;
      for (int64_t bit = 0; bit < nbits; ++bit) {
        mask += (_randbits[idx * nbits + bit] & 0x1) << bit;
      }
      x_xor_r[idx] = _x[idx] ^ mask;
    });

    // open c = x ^ r
    x_xor_r = comm->allReduce<U, std::bit_xor>(x_xor_r, "open(x^r)");

    MemRefView<U> _res(res);
    pforeach(0, numel, [&](int64_t idx) {
      _res[idx] = 0;
      for (int64_t bit = 0; bit < nbits; bit++) {
        auto c_i = (x_xor_r[idx] >> bit) & 0x1;
        if (comm->getRank() == 0) {
          _res[idx] += (c_i + (1 - c_i * 2) * _randbits[idx * nbits + bit])
                       << bit;
        } else {
          _res[idx] += ((1 - c_i * 2) * _randbits[idx * nbits + bit]) << bit;
        }
      }
    });
  });

  if (comm->getRank() == 2) {
    comm->sendAsync(0, res, "res");
    ring_zeros(res);
  }
  if (comm->getRank() == 0) {
    auto tmp = comm->recv(2, aty, "res");
    tmp = tmp.reshape(x.shape());
    res = ring_add(res, tmp);
  }
  return res;
}

MemRef Msb_a2b::proc(KernelEvalContext* ctx, const MemRef& in) const {
  // SC
  auto in_ = ring_add(in, in);
  in_ = ShareConvert().proc(ctx, in_);
  auto res = Msb().proc(ctx, in_);
  res = A2B().proc(ctx, res);
  return res;
}

void CommonTypeV::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  const auto* lhs_v = lhs.as<Priv2kTy>();
  const auto* rhs_v = rhs.as<Priv2kTy>();
  const auto valid_bits = std::max(lhs_v->valid_bits(), rhs_v->valid_bits());

  auto st = SizeOf(lhs_v->storage_type()) > SizeOf(rhs_v->storage_type())
                ? lhs_v->semantic_type()
                : rhs_v->semantic_type();
  ctx->pushOutput(makeType<ArithShareTy>(st, valid_bits));
}

MemRef RingCastS::proc(KernelEvalContext*, const MemRef& in,
                       SemanticType to_type) const {
  MemRef out(in);
  out.eltype().as<BaseRingType>()->set_semantic_type(to_type);
  return out;
}

std::vector<MemRef> BitDecompose::proc(KernelEvalContext* ctx,
                                       const MemRef& in) const {
  SPU_ENFORCE(in.eltype().as<BoolShare>());
  const auto* ty = in.eltype().as<BoolShareTy>();
  size_t nbits = ty->valid_bits();
  SPU_ENFORCE_GT(nbits, 0U);
  std::vector<MemRef> outs;
  auto bty = makeType<BoolShareTy>(ty->semantic_type(), ST_8, 1);
  DISPATCH_ALL_STORAGE_TYPES(ty->storage_type(), [&]() {
    using InT = ScalarT;
    MemRefView<InT> _in(in);
    DISPATCH_ALL_STORAGE_TYPES(bty.storage_type(), [&]() {
      using OutT = ScalarT;
      for (size_t i = 0; i < nbits; ++i) {
        MemRef bit(bty, in.shape());
        MemRefView<OutT> _bit(bit);
        pforeach(0, in.numel(), [&](int64_t idx) {
          _bit[idx] = static_cast<OutT>((_in[idx] >> i) & 0x1);
        });
        outs.push_back(std::move(bit));
      }
    });
  });
  return outs;
}

MemRef BitCompose::proc(KernelEvalContext* ctx,
                        const std::vector<MemRef>& in) const {
  SPU_ENFORCE(!in.empty());
  size_t nbits = in.size();
  auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto sst = GetStorageType(field);
  auto smt = in[0].eltype().semantic_type();
  const auto ty = makeType<BoolShareTy>(smt, sst, nbits);
  MemRef out(ty, in[0].shape());
  ring_zeros(out);
  DISPATCH_ALL_STORAGE_TYPES(sst, [&]() {
    using OutT = ScalarT;
    MemRefView<OutT> _out(out);
    for (size_t i = 0; i < nbits; ++i) {
      DISPATCH_ALL_STORAGE_TYPES(in[i].eltype().storage_type(), [&]() {
        using InT = ScalarT;
        MemRefView<InT> _in(in[i]);
        pforeach(0, in[0].numel(), [&](int64_t idx) {
          _out[idx] |= static_cast<OutT>(_in[idx] & 0x1) << i;
        });
      });
    }
  });
  return out;
}

}  // namespace spu::mpc::securenn
