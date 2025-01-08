// Copyright 2024 Ant Group Co., Ltd.
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
#include "prime_utils.h"

#include "type.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

NdArrayRef ProbConvRing2k(const NdArrayRef& inp_share, int rank,
                          size_t shr_width) {
  SPU_ENFORCE(inp_share.eltype().isa<RingTy>());
  SPU_ENFORCE(rank >= 0 && rank <= 1);

  auto eltype = inp_share.eltype();
  NdArrayRef output_share(eltype, inp_share.shape());

  auto ring_ty = eltype.as<RingTy>()->field();
  uint128_t shifted_bit = 1;
  shifted_bit <<= shr_width;
  auto mask = shifted_bit - 1;
  // x mod p - 1
  // in our case p > 2^shr_width

  DISPATCH_ALL_FIELDS(ring_ty, [&]() {
    const auto prime = ScalarTypeToPrime<ring2k_t>::prime;
    ring2k_t prime_minus_one = (prime - 1);
    NdArrayView<const ring2k_t> inp(inp_share);
    NdArrayView<ring2k_t> output_share_view(output_share);
    pforeach(0, output_share.numel(), [&](int64_t i) {
      output_share_view[i] =
          rank == 0 ? ((inp[i] & mask) % prime_minus_one)
                    // numerical considerations here
                    // we wanted to work on ring 2k or field p - 1
                    // however, if we do not add p -1
                    // then the computation will resort to int128
                    // due to the way computer works
                    : ((inp[i] & mask) + prime_minus_one - shifted_bit) %
                          prime_minus_one;
    });
  });
  return output_share;
}

NdArrayRef UnflattenBuffer(yacl::Buffer&& buf, const NdArrayRef& x) {
  return NdArrayRef(std::make_shared<yacl::Buffer>(std::move(buf)), x.eltype(),
                    x.shape());
}

// P0 holds x，P1 holds y
//  Beaver generates ab = c_0 + c_1
//  Give (a, c_0) to P0
//  Give (b, c_1) to P1
std::tuple<NdArrayRef, NdArrayRef> MulPrivPrep(KernelEvalContext* ctx,
                                               const NdArrayRef& x) {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  // generate beaver multiple triple.
  NdArrayRef a_or_b;
  NdArrayRef c;

  const size_t numel = x.shape().numel();
  auto [a_or_b_buf, c_buf] = beaver->MulPriv(
      field, numel,  //
      x.eltype().isa<GfmpTy>() ? ElementType::kGfmp : ElementType::kRing);
  SPU_ENFORCE(static_cast<size_t>(a_or_b_buf.size()) == numel * SizeOf(field));
  SPU_ENFORCE(static_cast<size_t>(c_buf.size()) == numel * SizeOf(field));

  a_or_b = UnflattenBuffer(std::move(a_or_b_buf), x);
  c = UnflattenBuffer(std::move(c_buf), x);

  return {std::move(a_or_b), std::move(c)};
}

// P0 holds x，P1 holds y
//  Beaver generates ab = c_0 + c_1
//  Give (a, c_0) to P0
//  Give (b, c_1) to P1
//
// - P0 sends (x+a) to P1 ; P1 sends (y+b) to P0
// - P0 calculates z0 = x(y+b) + c0 ; P1 calculates z1 = -b(x+a) + c1
NdArrayRef MulPriv(KernelEvalContext* ctx, const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<RingTy>());
  auto* comm = ctx->getState<Communicator>();

  NdArrayRef a_or_b, c, xa_or_yb;

  std::tie(a_or_b, c) = MulPrivPrep(ctx, x);

  // P0 sends (x+a) to P1 ; P1 sends (y+b) to P0
  comm->sendAsync(comm->nextRank(), ring_add(a_or_b, x), "(x + a) or (y + b)");
  xa_or_yb = comm->recv(comm->prevRank(), x.eltype(), "(x + a) or (y + b)")
                 .reshape(x.shape());
  // note that our rings are commutative.
  if (comm->getRank() == 0) {
    ring_add_(c, ring_mul(std::move(xa_or_yb), x));
  }
  if (comm->getRank() == 1) {
    ring_sub_(c, ring_mul(std::move(xa_or_yb), a_or_b));
  }
  return c;
}

NdArrayRef MulPrivModMP(KernelEvalContext* ctx, const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<GfmpTy>());
  auto* comm = ctx->getState<Communicator>();

  NdArrayRef a_or_b, c, xa_or_yb;
  std::tie(a_or_b, c) = MulPrivPrep(ctx, x);

  comm->sendAsync(comm->nextRank(), gfmp_add_mod(a_or_b, x), "xa_or_yb");
  xa_or_yb =
      comm->recv(comm->prevRank(), x.eltype(), "xa_or_yb").reshape(x.shape());

  // note that our rings are commutative.
  if (comm->getRank() == 0) {
    gfmp_add_mod_(c, gfmp_mul_mod(std::move(xa_or_yb), x));
  }
  if (comm->getRank() == 1) {
    gfmp_sub_mod_(c, gfmp_mul_mod(std::move(xa_or_yb), a_or_b));
  }
  return c;
}

// We assume the input is ``positive''
// Given h0 + h1 = h mod p and h < p / 2
// Define b0 = 1{h0 >= p/2}
//        b1 = 1{h1 >= p/2}
// Compute w = 1{h0 + h1 >= p}
// It can be proved that w = (b0 or b1) = not (not b0 and not b1)
NdArrayRef WrapBitModMP(KernelEvalContext* ctx, const NdArrayRef& x) {
  // create a wrap bit NdArrayRef of the same shape as in
  NdArrayRef b(x.eltype(), x.shape());

  // for each element, we compute b = 1{h < p/2} for each private share piece
  const auto numel = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(field, [&]() {
    ring2k_t prime = ScalarTypeToPrime<ring2k_t>::prime;
    ring2k_t phalf = prime >> 1;
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _b(b);
    pforeach(0, numel, [&](int64_t idx) {
      _b[idx] = static_cast<ring2k_t>(_x[idx] < phalf);
    });

    // do private mul
    b = MulPriv(ctx, b.as(makeType<RingTy>(field)));

    // map 1 to 0 and 0 to 1, use 1 - x
    if (ctx->getState<Communicator>()->getRank() == 0) {
      pforeach(0, numel, [&](int64_t idx) { _b[idx] = 1 - _b[idx]; });
    } else {
      pforeach(0, numel, [&](int64_t idx) { _b[idx] = -_b[idx]; });
    }
  });

  return b;
}
// Mersenne Prime share -> Ring2k share

NdArrayRef ConvMP(KernelEvalContext* ctx, const NdArrayRef& h,
                  uint truncate_nbits) {
  // calculate wrap bit
  NdArrayRef w = WrapBitModMP(ctx, h);
  const auto field = h.eltype().as<Ring2k>()->field();
  const auto numel = h.numel();

  // x = (h - p * w) mod 2^k

  NdArrayRef x(makeType<RingTy>(field), h.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto prime = ScalarTypeToPrime<ring2k_t>::prime;
    NdArrayView<ring2k_t> h_view(h);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> w_view(w);
    pforeach(0, numel, [&](int64_t idx) {
      _x[idx] = static_cast<ring2k_t>(h_view[idx] >> truncate_nbits) -
                static_cast<ring2k_t>(prime >> truncate_nbits) * w_view[idx];
    });
  });
  return x;
}

}  // namespace spu::mpc::semi2k
