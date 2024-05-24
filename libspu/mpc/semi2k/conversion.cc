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

#include "libspu/mpc/semi2k/conversion.h"

#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_a2b(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(a2b(ctx, WrapValue(x)));
}

static NdArrayRef wrap_and_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(and_bb(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<NdArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] =
        prg_state->genPrssPair(field, x.shape(), PrgState::GenPrssCtrl::Both);
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
  return r_a;
}

NdArrayRef B2A_Randbit::proc(KernelEvalContext* ctx,
                             const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const int64_t nbits = x.eltype().as<BShare>()->nbits();
  SPU_ENFORCE((size_t)nbits <= SizeOf(field) * 8, "invalid nbits={}", nbits);
  if (nbits == 0) {
    // special case, it's known to be zero.
    return ring_zeros(field, x.shape()).as(makeType<AShrTy>(field));
  }

  const auto numel = x.numel();
  const auto rand_numel = numel * static_cast<int64_t>(nbits);

  auto randbits = beaver->RandBit(field, rand_numel);
  SPU_ENFORCE(static_cast<size_t>(randbits.size()) ==
              rand_numel * SizeOf(field));
  auto res = NdArrayRef(makeType<AShrTy>(field), x.shape());

  DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
    using U = ring2k_t;

    absl::Span<const U> _randbits(randbits.data<U>(), rand_numel);
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

  return res;
}

NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // For if k > 2 parties does not collude with each other, then we can
  // construct two additive share and use carray out circuit directly.
  SPU_ENFORCE(comm->getWorldSize() == 2, "only support for 2PC, got={}",
              comm->getWorldSize());

  std::vector<NdArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] =
        prg_state->genPrssPair(field, in.shape(), PrgState::GenPrssCtrl::Both);
    auto b = ring_xor(r0, r1).as(bty);
    if (idx == comm->getRank()) {
      ring_xor_(b, in);
    }
    bshrs.push_back(b.as(bty));
  }

  // Compute the k-1'th carry bit.
  size_t k = SizeOf(field) * 8 - 1;
  if (in.numel() == 0) {
    k = 0;  // Empty matrix
  }

  auto* sctx = ctx->sctx();
  const Shape shape = {in.numel()};
  auto m = WrapValue(bshrs[0]);
  auto n = WrapValue(bshrs[1]);
  {
    auto carry = carry_a2b(sctx, m, n, k);

    // Compute the k'th bit.
    //   (m^n)[k] ^ carry
    auto msb = xor_bb(sctx, rshift_b(sctx, xor_bb(sctx, m, n), k), carry);

    return UnwrapValue(msb);
  }
}

NdArrayRef eqz(KernelEvalContext* ctx, const NdArrayRef& in) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();

  NdArrayRef out(makeType<BShrTy>(field), in.shape());

  size_t pivot;
  prg_state->fillPubl(absl::MakeSpan(&pivot, 1));
  pivot %= comm->getWorldSize();
  // beaver samples r and deals [r]a and [r]b
  //  receal c = a+r
  // check a == 0  <=> c == r
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = ring2k_t;
    auto [ra_buf, rb_buf] = beaver->Eqz(field, numel);

    NdArrayRef rb(std::make_shared<yacl::Buffer>(std::move(rb_buf)),
                  in.eltype(), in.shape());
    {
      NdArrayRef c_p;
      {
        NdArrayRef ra(std::make_shared<yacl::Buffer>(std::move(ra_buf)),
                      in.eltype(), in.shape());
        // c in secret share
        ring_add_(ra, in);
        // reveal c
        c_p = comm->allReduce(ReduceOp::ADD, ra, "reveal c ");
      }

      if (comm->getRank() == pivot) {
        ring_xor_(rb, c_p);
        ring_not_(rb);
      }
    }

    // if a == 0, ~(a+ra) ^ rb supposed to be all 1
    // do log(k) round bit wise and
    // TODO: fix AND triple
    // in beaver->AND(field, shape), min FM32, need min 1byte to reduce comm
    NdArrayRef round_out = rb.as(makeType<BShrTy>(field));
    size_t cur_bits = round_out.eltype().as<BShare>()->nbits();
    while (cur_bits != 1) {
      cur_bits /= 2;
      round_out =
          wrap_and_bb(ctx->sctx(), round_out, ring_rshift(round_out, cur_bits));
    }

    // 1 bit info in lsb
    NdArrayView<el_t> _out(out);
    NdArrayView<el_t> _round_out(round_out);
    pforeach(0, numel, [&](int64_t idx) { _out[idx] = _round_out[idx] & 1; });
  });

  return out;
}

NdArrayRef EqualAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

  out = ring_sub(lhs, rhs);

  return eqz(ctx, out);
}

NdArrayRef EqualAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

  auto rank = comm->getRank();
  if (rank == 0) {
    out = ring_sub(lhs, rhs);
  } else {
    out = lhs;
  };

  return eqz(ctx, out);
}

void CommonTypeV::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  const auto* lhs_v = lhs.as<Priv2kTy>();
  const auto* rhs_v = rhs.as<Priv2kTy>();

  ctx->setOutput(makeType<AShrTy>(std::max(lhs_v->field(), rhs_v->field())));
}

}  // namespace spu::mpc::semi2k
