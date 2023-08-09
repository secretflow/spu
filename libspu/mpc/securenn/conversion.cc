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

#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/securenn/arithmetic.h"
#include "libspu/mpc/securenn/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::securenn {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_a2b(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(a2b(ctx, WrapValue(x)));
}

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<NdArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] = prg_state->genPrssPair(field, x.shape());
    auto b = ring_xor(r0, r1).as(bty);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }
    bshrs.push_back(b.as(bty));
  }

  NdArrayRef res = vreduce(bshrs.begin(), bshrs.end(),
                           [&](const NdArrayRef& xx, const NdArrayRef& yy) {
                             return wrap_add_bb(ctx->sctx(), xx, yy);
                           });
  return res.as(bty);
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  auto r_v = prg_state->genPriv(field, x.shape());
  auto r_a = r_v.as(makeType<AShrTy>(field));

  // convert r to boolean share.
  auto r_b = wrap_a2b(ctx->sctx(), r_a);

  // evaluate adder circuit on x & r, and reveal x+r
  auto x_plus_r = comm->allReduce(ReduceOp::XOR,
                                  wrap_add_bb(ctx->sctx(), x, r_b), kBindName);

  // compute -r + (x+r)
  ring_neg_(r_a);
  if (comm->getRank() == 0) {
    ring_add_(r_a, x_plus_r);
  }
  if (comm->getRank() == 2) {
    comm->sendAsync(0, r_a, "r_a");
    r_a = ring_zeros(field, x.shape());
  }
  if (comm->getRank() == 0) {
    auto tmp = comm->recv(2, makeType<AShrTy>(field), "r_a");
    tmp = tmp.reshape(x.shape());
    r_a = ring_add(r_a, tmp);
  }
  return r_a;
}

NdArrayRef B2A_Randbit::proc(KernelEvalContext* ctx,
                             const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto rank = comm->getRank();

  const int64_t nbits = x.eltype().as<BShare>()->nbits();
  SPU_ENFORCE((size_t)nbits <= SizeOf(field) * 8, "invalid nbits={}", nbits);
  if (nbits == 0) {
    // special case, it's known to be zero.
    return ring_zeros(field, x.shape()).as(makeType<AShrTy>(field));
  }

  auto numel = x.numel();

  auto randbits =
      prg_state->genPriv(field, {numel * static_cast<int64_t>(nbits)});
  // reconstruct ranbits
  if (rank == 0) comm->sendAsync(2, randbits, "randbits0");
  if (rank == 1) comm->sendAsync(2, randbits, "randbits1");
  if (rank == 2) {
    auto randbits0 = comm->recv(0, makeType<AShrTy>(field), "randbits0");
    randbits0 = randbits0.reshape(randbits.shape());
    auto randbits1 = comm->recv(1, makeType<AShrTy>(field), "randbits1");
    randbits1 = randbits1.reshape(randbits.shape());
    auto randbits_recon = ring_add(ring_add(randbits, randbits0), randbits1);
    auto adjust =
        ring_sub(ring_randbit(field, randbits.shape()), randbits_recon);
    comm->sendAsync(0, adjust, "adjust");
  }
  if (rank == 0) {
    auto adjust = comm->recv(2, makeType<AShrTy>(field), "adjust");
    adjust = adjust.reshape(randbits.shape());
    ring_add_(randbits, adjust);
  }

  auto res = NdArrayRef(makeType<AShrTy>(field), x.shape());

  DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
    using U = ring2k_t;

    NdArrayView<U> _randbits(randbits);
    NdArrayView<U> _x(x);

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

    NdArrayView<U> _res(res);
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
    res = ring_zeros(field, x.shape()).as(makeType<AShrTy>(field));
  }
  if (comm->getRank() == 0) {
    auto tmp = comm->recv(2, makeType<AShrTy>(field), "res");
    tmp = tmp.reshape(x.shape());
    res = ring_add(res, tmp);
  }
  return res;
}

NdArrayRef Msb_a2b::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  // SC
  // auto in_ = ring_add(in, in);
  // auto in_ = in;
  // auto in_ = ShareConvert().proc(ctx, in);
  auto res = Msb_opt().proc(ctx, in);
  // auto res = Msb().proc(ctx, in_);
  res = A2B().proc(ctx, res);
  return res;
}

}  // namespace spu::mpc::securenn
