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
#include "exp.h"

#include "prime_utils.h"
#include "type.h"

#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

// Given [x*2^fxp] mod 2k for x
// compute [exp(x) * 2^fxp] mod 2^k

// Assume x is in valid range, otherwise the error may be too large to
// use this method.

NdArrayRef ExpA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const size_t fxp = ctx->sctx()->getFxpBits();
  SPU_ENFORCE(
      fxp < 64,
      "fxp must be less than 64 for this method, or shift bit overflow ",
      "may occur");
  auto field = in.eltype().as<Ring2k>()->field();
  NdArrayRef x = in.clone();
  NdArrayRef out;

  // TODO: set different values for FM64 FM32
  const size_t kExpFxp = (field == FieldType::FM128) ? 24 : 13;

  const int rank = ctx->sctx()->lctx()->Rank();
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto total_fxp = kExpFxp + fxp;
    // note that x is already encoded with fxp
    // this conv scale further converts x int fixed point numbers with
    // total_fxp
    const ring2k_t exp_conv_scale = std::roundf(M_LOG2E * (1L << kExpFxp));

    // offset scale should directly encoded to a fixed point with total_fxp
    const ring2k_t offset =
        ctx->sctx()->config().experimental_exp_prime_offset();
    const ring2k_t offset_scaled = offset << total_fxp;

    NdArrayView<ring2k_t> _x(x);
    if (rank == 0) {
      pforeach(0, x.numel(), [&](ring2k_t i) {
        _x[i] *= exp_conv_scale;
        _x[i] += offset_scaled;
      });
    } else {
      pforeach(0, x.numel(), [&](ring2k_t i) { _x[i] *= exp_conv_scale; });
    }
    size_t shr_width = SizeOf(field) * 8 - fxp;

    const ring2k_t kBit = 1;
    auto shifted_bit = kBit << total_fxp;
    const ring2k_t frac_mask = shifted_bit - 1;

    auto int_part = ring_arshift(x, {static_cast<int64_t>(total_fxp)});

    // convert from ring-share (int-part) to a prime share over p - 1
    int_part = ProbConvRing2k(int_part, rank, shr_width);
    NdArrayView<ring2k_t> int_part_view(int_part);

    pforeach(0, x.numel(), [&](int64_t i) {
      // y = 2^int_part mod p
      ring2k_t y = exp_mod<ring2k_t>(2, int_part_view[i]);
      // z = 2^fract_part in RR
      double frac_part = static_cast<double>(_x[i] & frac_mask) / shifted_bit;
      frac_part = std::pow(2., frac_part);

      // Multiply the 2^{int_part} * 2^{frac_part} mod p
      // note that mul_mod uses mersenne prime as modulus according to field
      int_part_view[i] = mul_mod<ring2k_t>(
          y, static_cast<ring2k_t>(std::roundf(frac_part * (kBit << fxp))));
    });

    NdArrayRef muled = MulPrivModMP(ctx, int_part.as(makeType<GfmpTy>(field)));

    out = ConvMP(ctx, muled, offset + fxp);
  });
  return out.as(in.eltype());
}

}  // namespace spu::mpc::semi2k