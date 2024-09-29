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
  return r_a;
}

// TODO(jimi): pack {numel * nbits} to fully make use of undelying storage to
// save communications. If implemented, B2A_Disassemble kernel is also no longer
// needed
MemRef B2A_Randbit::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const auto* ty = x.eltype().as<BoolShareTy>();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const int64_t nbits = ty->valid_bits();
  const int64_t field = SizeOf(ty->storage_type()) * 8;
  if (nbits == 0) {
    // special case, it's known to be zero.
    MemRef ret(makeType<ArithShareTy>(ty->semantic_type(), field), x.shape());
    ring_zeros(ret);
    return ret;
  }

  const auto numel = x.numel();
  const auto backtype = GetStorageType(nbits);
  const auto rand_numel = numel * static_cast<int64_t>(nbits);

  auto randbits = beaver->RandBit(field, rand_numel);
  auto res =
      MemRef(makeType<ArithShareTy>(ty->semantic_type(), field), x.shape());
  SPU_ENFORCE(static_cast<size_t>(randbits.size()) ==
              rand_numel * SizeOf(field));

  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using U = ScalarT;

    absl::Span<const U> _randbits(randbits.data<U>(), rand_numel);
    MemRefView<U> _x(x);

    // algorithm begins.
    // Ref: III.D @ https://eprint.iacr.org/2019/599.pdf (SPDZ-2K primitives)
    DISPATCH_ALL_STORAGE_TYPES(backtype, [&]() {
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

      MemRefView<U> _res(res);
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
std::vector<MemRef> B2A_Disassemble::proc(KernelEvalContext* ctx,
                                          const MemRef& x) const {
  const auto* ty = x.eltype().as<BoolShareTy>();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const int64_t nbits = ty->valid_bits();
  const int64_t field = SizeOf(ty->storage_type()) * 8;

  const auto numel = x.numel();
  const auto rand_numel = numel * static_cast<int64_t>(nbits);
  const auto backtype = GetStorageType(nbits);

  auto randbits = beaver->RandBit(field, rand_numel);

  std::vector<MemRef> res;
  res.reserve(nbits);
  for (int64_t idx = 0; idx < nbits; ++idx) {
    res.emplace_back(makeType<ArithShareTy>(ty->semantic_type(), field),
                     x.shape());
  }
  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using U = ScalarT;

    absl::Span<const U> _randbits(randbits.data<U>(), rand_numel);
    MemRefView<U> _x(x);

    DISPATCH_ALL_STORAGE_TYPES(backtype, [&]() {
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
          MemRefView<U> _res(res[bit]);
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

  return res;
}

MemRef MsbA2B::proc(KernelEvalContext* ctx, const MemRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // For if k > 2 parties does not collude with each other, then we can
  // construct two additive share and use carry out circuit directly.
  SPU_ENFORCE(comm->getWorldSize() == 2, "only support for 2PC, got={}",
              comm->getWorldSize());

  std::vector<MemRef> bshrs;
  const auto bty = makeType<BoolShareTy>(
      SE_1, in.eltype().storage_type(), SizeOf(in.eltype().storage_type()) * 8);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    MemRef r0(in.eltype(), in.shape());
    MemRef r1(in.eltype(), in.shape());
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());

    auto b = ring_xor(r0, r1).as(bty);

    if (idx == comm->getRank()) {
      ring_xor_(b, in);
    }
    bshrs.push_back(b.as(bty));
  }

  // Compute the k-1'th carry bit.
  size_t k = SizeOf(in.eltype().storage_type()) * 8 - 1;
  if (in.numel() == 0) {
    k = 0;  // Empty matrix
  }

  auto* sctx = ctx->sctx();
  const Shape shape = {in.numel()};
  auto m = bshrs[0];
  auto n = bshrs[1];
  {
    auto carry = carry_a2b(sctx, m, n, k);

    // Compute the k'th bit.
    //   (m^n)[k] ^ carry
    auto msb = xor_bb(
        sctx, rshift_b(sctx, xor_bb(sctx, m, n), {static_cast<int64_t>(k)}),
        carry);

    return msb;
  }
}

MemRef eqz(KernelEvalContext* ctx, const MemRef& in) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const auto field = in.eltype().as<ArithShareTy>()->valid_bits();
  const auto numel = in.numel();

  MemRef out(makeType<BoolShareTy>(SE_1, in.eltype().storage_type(), field),
             in.shape());

  size_t pivot;
  prg_state->fillPubl(&pivot, sizeof(size_t));
  pivot %= comm->getWorldSize();
  // beaver samples r and deals [r]a and [r]b
  //  receal c = a+r
  // check a == 0  <=> c == r
  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using el_t = ScalarT;
    auto [ra_buf, rb_buf] = beaver->Eqz(field, numel);

    MemRef rb(std::make_shared<yacl::Buffer>(std::move(rb_buf)), in.eltype(),
              in.shape());
    {
      MemRef c_p;
      {
        MemRef ra(std::make_shared<yacl::Buffer>(std::move(ra_buf)),
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
    MemRef round_out = rb.as(makeType<BoolShareTy>(
        in.eltype().semantic_type(), in.eltype().storage_type(), field));
    int64_t cur_bits = round_out.eltype().as<BoolShareTy>()->valid_bits();
    while (cur_bits != 1) {
      cur_bits /= 2;
      round_out =
          and_bb(ctx->sctx(), round_out, ring_rshift(round_out, {cur_bits}));
    }

    // 1 bit info in lsb
    MemRefView<el_t> _out(out);
    MemRefView<el_t> _round_out(round_out);
    pforeach(0, numel, [&](int64_t idx) { _out[idx] = _round_out[idx] & 1; });
  });

  return out;
}

MemRef EqualAA::proc(KernelEvalContext* ctx, const MemRef& lhs,
                     const MemRef& rhs) const {
  SPU_ENFORCE_EQ(lhs.eltype(), rhs.eltype());
  MemRef out(lhs.eltype(), lhs.shape());

  out = ring_sub(lhs, rhs);

  return eqz(ctx, out);
}

MemRef EqualAP::proc(KernelEvalContext* ctx, const MemRef& lhs,
                     const MemRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();

  MemRef out(lhs.eltype(), lhs.shape());

  auto rank = comm->getRank();
  if (rank == 0) {
    if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
      MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                       SizeOf(lhs.eltype().storage_type()) * 8),
                      rhs.shape());
      ring_assign(rhs_cast, rhs);
      out = ring_sub(lhs, rhs_cast).as(lhs.eltype());
    } else {
      out = ring_sub(lhs, rhs);
    }
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

}  // namespace spu::mpc::semi2k
