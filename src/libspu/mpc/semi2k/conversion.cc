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

// TODO: Move to some common place
PtType getBacktype(size_t nbits) {
  if (nbits <= 8) {
    return PT_U8;
  }
  if (nbits <= 16) {
    return PT_U16;
  }
  if (nbits <= 32) {
    return PT_U32;
  }
  if (nbits <= 64) {
    return PT_U64;
  }
  if (nbits <= 128) {
    return PT_U128;
  }
  SPU_THROW("invalid number of bits={}", nbits);
}

namespace {
NdArrayRef a2b_impl(KernelEvalContext* ctx, const NdArrayRef& x,
                    int64_t nbits) {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<NdArrayRef> bshrs;
  int64_t valid_bits = nbits == -1 ? SizeOf(field) * 8 : nbits;

  const auto bty = makeType<BShrTy>(field, valid_bits);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] =
        prg_state->genPrssPair(field, x.shape(), PrgState::GenPrssCtrl::Both);
    auto b = ring_xor(r0, r1);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }

    bshrs.push_back(b.as(bty));
  }

  NdArrayRef res = vreduce(bshrs.begin(), bshrs.end(),
                           [&](const NdArrayRef& xx, const NdArrayRef& yy) {
                             return wrap_add_bb(ctx->sctx(), xx, yy);
                           });

  if (nbits != -1) {
    ring_bitmask_(res, 0, valid_bits);
  }
  return res.as(bty);
}
}  // namespace

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  // full bits A2B
  return a2b_impl(ctx, x, -1);
}

NdArrayRef A2B_Bits::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          int64_t nbits) const {
  return a2b_impl(ctx, x, nbits);
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
  auto x_plus_r = comm->allReduce(
      ReduceOp::XOR, wrap_add_bb(ctx->sctx(), x, r_b), kBindName());

  // compute -r + (x+r)
  ring_neg_(r_a);
  if (comm->getRank() == 0) {
    ring_add_(r_a, x_plus_r);
  }
  return r_a;
}

// TODO(jimi): pack {numel * nbits} to fully make use of undelying storage to
// save communications. If implemented, B2A_Disassemble kernel is also no longer
// needed
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
  const PtType backtype = getBacktype(nbits);

  auto randbits = beaver->RandBit(field, rand_numel);
  SPU_ENFORCE(static_cast<size_t>(randbits.size()) ==
              rand_numel * SizeOf(field));
  auto res = NdArrayRef(makeType<AShrTy>(field), x.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using U = ring2k_t;

    absl::Span<const U> _randbits(randbits.data<U>(), rand_numel);
    NdArrayView<U> _x(x);

    // algorithm begins.
    // Ref: III.D @ https://eprint.iacr.org/2019/599.pdf (SPDZ-2K primitives)
    DISPATCH_UINT_PT_TYPES(backtype, [&]() {
      using V = ScalarT;
      std::vector<V> x_xor_r(numel);

      pforeach(0, numel, [&](int64_t idx) {
        // use _r[i*nbits, (i+1)*nbits) to construct rb[i]
        V mask = 0;
        for (int64_t bit = 0; bit < nbits; ++bit) {
          mask += (static_cast<V>(_randbits[idx * nbits + bit]) & 0x1) << bit;
        }
        x_xor_r[idx] = _x[idx] ^ mask;
      });

      // open c = x ^ r
      x_xor_r = comm->allReduce<V, std::bit_xor>(x_xor_r, "open(x^r)");

      NdArrayView<U> _res(res);
      pforeach(0, numel, [&](int64_t idx) {
        _res[idx] = 0;
        for (int64_t bit = 0; bit < nbits; bit++) {
          auto c_i = static_cast<U>(x_xor_r[idx] >> bit) & 0x1;
          if (comm->getRank() == 0) {
            _res[idx] += (c_i + (1 - c_i * 2) * _randbits[idx * nbits + bit])
                         << bit;
          } else {
            _res[idx] += ((1 - c_i * 2) * _randbits[idx * nbits + bit]) << bit;
          }
        }
      });
    });
  });

  return res;
}

// Reference:
//  III.D @ https://eprint.iacr.org/2019/599.pdf (SPDZ-2K primitives)
//
// Analysis:
//  Online Latency: 1 (x_xor_r reveal)
//  Communication: one element bits for one element
//  Vectorization: yes
//
// HighLevel Intuition:
//  Since: X = sum: Xi * 2^i
// If we have <Xi>A, then we can construct <X>A = sum: <Xi>A * 2^i.
//
// The problem is that we only have <Xi>B in hand. Details for how to
// construct <Xi>A from <Xi>B:
// - trusted third party choose a random bit r, where r == 0 or r == 1.
// - trusted third party send <r>A to parties
// - parties compute <r>B from <r>A
// - parties xor_open c = Xi ^ r = open(<Xi>B ^ <r>B), Xi is still safe due
// to protection from r.
// - parties compute: <x> = c + (1-2c)*<r>
//    <Xi>A = 1 - <r>A if c == 1, i.e. Xi != r
//    <Xi>A = <r>A if c == 0, i.e. Xi == r
//    i.e. <Xi>A = c + (1-2c) * <r>A
//
//  Online Communication:
//    = 1 (xor open)

// Disassemble BShr to AShr bit-by-bit
//  Input: BShr
//  Return: a vector of k AShr, k is the valid bits of BShr
std::vector<NdArrayRef> B2A_Disassemble::proc(KernelEvalContext* ctx,
                                              const NdArrayRef& x,
                                              FieldType perm_field) const {
  const auto bshr_field = x.eltype().as<Ring2k>()->field();
  FieldType ashr_field = bshr_field;
  if (perm_field != FT_INVALID) {
    ashr_field = perm_field;
  }
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const int64_t nbits = x.eltype().as<BShare>()->nbits();
  SPU_ENFORCE((size_t)nbits > 0 && (size_t)nbits <= SizeOf(bshr_field) * 8,
              "invalid nbits={}", nbits);

  const auto numel = x.numel();
  const auto rand_numel = numel * static_cast<int64_t>(nbits);
  const PtType backtype = getBacktype(nbits);

  auto randbits = beaver->RandBit(ashr_field, rand_numel);

  std::vector<NdArrayRef> res;
  res.reserve(nbits);
  for (int64_t idx = 0; idx < nbits; ++idx) {
    res.emplace_back(makeType<AShrTy>(ashr_field), x.shape());
  }
  DISPATCH_ALL_FIELDS(ashr_field, [&]() {
    using U = ring2k_t;

    DISPATCH_ALL_FIELDS(bshr_field, [&]() {
      using UU = ring2k_t;

      absl::Span<const U> _randbits(randbits.data<U>(), rand_numel);
      NdArrayView<UU> _x(x);

      DISPATCH_UINT_PT_TYPES(backtype, [&]() {
        using V = ScalarT;
        std::vector<V> x_xor_r(numel);

        pforeach(0, numel, [&](int64_t idx) {
          // use _r[i*nbits, (i+1)*nbits) to construct rb[i]
          V mask = 0;
          for (int64_t bit = 0; bit < nbits; ++bit) {
            mask += (static_cast<V>(_randbits[idx * nbits + bit]) & 0x1) << bit;
          }
          x_xor_r[idx] = _x[idx] ^ mask;
        });

        // open c = x ^ r
        x_xor_r = comm->allReduce<V, std::bit_xor>(x_xor_r, "open(x^r)");

        pforeach(0, numel, [&](int64_t idx) {
          pforeach(0, nbits, [&](int64_t bit) {
            NdArrayView<U> _res(res[bit]);
            auto c_i = static_cast<U>(x_xor_r[idx] >> bit) & 0x1;
            if (comm->getRank() == 0) {
              _res[idx] = (c_i + (1 - c_i * 2) * _randbits[idx * nbits + bit]);
            } else {
              _res[idx] = ((1 - c_i * 2) * _randbits[idx * nbits + bit]);
            }
          });
        });
      });
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
    auto msb = xor_bb(
        sctx, rshift_b(sctx, xor_bb(sctx, m, n), {static_cast<int64_t>(k)}),
        carry);

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
  DISPATCH_ALL_FIELDS(field, [&]() {
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
    int64_t cur_bits = round_out.eltype().as<BShare>()->nbits();
    while (cur_bits != 1) {
      cur_bits /= 2;
      round_out = wrap_and_bb(ctx->sctx(), round_out,
                              ring_rshift(round_out, {cur_bits}));
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

  ctx->pushOutput(makeType<AShrTy>(std::max(lhs_v->field(), rhs_v->field())));
}

}  // namespace spu::mpc::semi2k
