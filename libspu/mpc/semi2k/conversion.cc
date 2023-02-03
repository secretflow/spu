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

// TODO: remove this
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/ab_kernels.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/semi2k/object.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/util/ring_ops.h"

namespace spu::mpc::semi2k {

ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_TRACE_MPC_LEAF(ctx, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();

  std::vector<ArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto b = zero_b(ctx->caller(), x.numel());
    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }
    bshrs.push_back(b.as(bty));
  }

  ArrayRef res = vectorizedReduce(bshrs.begin(), bshrs.end(),
                                  [&](const ArrayRef& xx, const ArrayRef& yy) {
                                    return add_bb(ctx->caller(), xx, yy);
                                  });
  return res.as(makeType<BShrTy>(field));
}

ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_TRACE_MPC_LEAF(ctx, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  auto r_v = prg_state->genPriv(field, x.numel());
  auto r_a = r_v.as(makeType<AShrTy>(field));

  // convert r to boolean share.
  auto r_b = a2b(ctx->caller(), r_a);

  // evaluate adder circuit on x & r, and reveal x+r
  auto x_plus_r =
      comm->allReduce(ReduceOp::XOR, and_bb(ctx->caller(), x, r_b), kBindName);

  // compute -r + (x+r)
  ring_neg_(r_a);
  if (comm->getRank() == 0) {
    ring_add_(r_a, x_plus_r);
  }
  return r_a;
}

ArrayRef B2A_Randbit::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_TRACE_MPC_LEAF(ctx, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const size_t numel = x.numel();
  const size_t nbits = x.eltype().as<BShare>()->nbits();
  YACL_ENFORCE(nbits <= SizeOf(field) * 8, "invalid nbits={}", nbits);
  if (nbits == 0) {
    // special case, it's known to be zero.
    return ring_zeros(field, numel).as(makeType<AShrTy>(field));
  }

  auto randbits = beaver->RandBit(field, x.numel() * nbits);
  auto res = ArrayRef(makeType<AShrTy>(field), numel);

  DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
    auto xb = xt_adapt<ring2k_t>(x);
    auto ya = xt_mutable_adapt<ring2k_t>(res);
    auto ra = xt_adapt<ring2k_t>(randbits);

    auto open = [&](const auto& ss) -> xt::xarray<ring2k_t> {
      // return AllReduce(comm->lctx_, ReduceOp::XOR, ss, "_");
      // TODO: there are too many memory copies.
      return xt_adapt<ring2k_t>(comm->allReduce(
          ReduceOp::XOR, xt_to_array(ss, makeType<RingTy>(field)), "open"));
    };

    // algorithm begins.
    // Ref: III.D @ https://eprint.iacr.org/2019/599.pdf (SPDZ-2K primitives)
    xt::xarray<ring2k_t> rb = xt::zeros<ring2k_t>({numel});
    for (size_t i = 0; i < nbits; i++) {
      auto ra_i = xt::view(ra, xt::range(i * numel, (i + 1) * numel));
      // randbit A2B, leave lsb only, then pack it.
      rb += (ra_i & 0x1) << i;
    }

    // open c = x ^ r
    auto c = open(xb ^ rb);

    ya = xt::zeros<ring2k_t>({numel});
    for (size_t i = 0; i < nbits; i++) {
      auto ra_i = xt::view(ra, xt::range(i * numel, (i + 1) * numel));
      auto c_i = (c >> i) & 0x1;

      // compute c + (1 - 2*c) * <r>, then stack it.
      if (comm->getRank() == 0) {
        ya += (c_i + (1 - c_i * 2) * ra_i) << i;
      } else {
        ya += ((1 - c_i * 2) * ra_i) << i;
      }
    }
  });

  return res;
}

ArrayRef MsbA2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<AShrTy>()->field();
  auto* comm = ctx->getState<Communicator>();

  // For if k > 2 parties does not collude with each other, then we can
  // construct two additive share and use carray out circuit directly.
  YACL_ENFORCE(comm->getWorldSize() == 2, "only support for 2PC, got={}",
               comm->getWorldSize());

  std::vector<ArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto b = zero_b(ctx->caller(), in.numel());
    if (idx == comm->getRank()) {
      ring_xor_(b, in);
    }
    bshrs.push_back(b.as(bty));
  }

  // Compute the k-1'th carry bit.
  const size_t k = SizeOf(field) * 8 - 1;

  auto* obj = ctx->caller();
  ArrayRef carry = common::carry_out(obj, bshrs[0], bshrs[1], k);
  return xor_bb(obj, rshift_b(obj, xor_bb(obj, bshrs[0], bshrs[1]), k), carry);
}

}  // namespace spu::mpc::semi2k
