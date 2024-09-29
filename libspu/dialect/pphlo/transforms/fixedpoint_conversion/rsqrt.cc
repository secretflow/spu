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

#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/approximations.h"
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/builder.h"

namespace mlir::spu::pphlo::fixedpoint::impl {

// Reference:
// 1. https://eprint.iacr.org/2012/405.pdf, section 6.1
// 2.
// https://github.com/tf-encrypted/tf-encrypted/blob/3b0f14d26e900caf12a92a9ea2284ccd4d58e683/tf_encrypted/protocol/aby3/fp.py#L35-L52
// Goldschmidt iteration, needs an initial approximation of sqrt_inv(x).
// In the end, g is an approximation of sqrt(x) while h is an approximation of
// 1 / (2 * sqrt(x)).
Value sqrt_with_rsqrt(builder::FxpBuilder &builder, Value input) {
  auto c0 = builder.fxp_constant(0.5F);
  auto c1 = builder.fxp_constant(1.5F);

  auto y0 = rsqrt_approx(builder, input);
  auto g = builder.mul(input, y0);
  auto h = builder.mul(y0, c0);

  // iterations of 1 is enough.
  const int iterations = 1;

  for (int i = 0; i < iterations; i++) {
    auto gh = builder.mul(g, h);
    auto r = builder.substract(c1, gh);
    g = builder.mul(g, r);
    h = builder.mul(h, r);
  }

  return g;
}

Value rsqrt_np2(builder::FxpBuilder &builder, Value x) {
  // let e = NP2(x), z = 2^(e+f)
  auto h1b = highestOneBit(builder, x);
  return builder.lshift(h1b, 1);
}

Value rsqrt_init_guess(builder::FxpBuilder &builder, Value x, Value z) {
  auto fxp_bits = builder.getCurrentFxpBits();
  auto z_rev = builder.bitrev(z, 0, 2 * fxp_bits);

  auto u = builder.mul(x, builder.bitcast(z_rev, x.getType()));

  llvm::SmallVector<double, 4> coeffs;
  if (builder.getConfig().lower_accuracy_rsqrt) {
    coeffs = {0.0F, -5.9417F, 4.7979F};
    auto r =
        polynomial(builder, u, coeffs, SignType::Positive, SignType::Positive);
    return builder.add(r, builder.fxp_constant(3.1855F));
  } else {
    coeffs = {0.0F, -15.47994394F, 38.4714796F, -49.86605845F, 26.02942339F};
    auto r =
        polynomial(builder, u, coeffs, SignType::Positive, SignType::Positive);
    return builder.add(r, builder.fxp_constant(4.14285016F));
  }
}

Value rsqrt_comp(builder::FxpBuilder &builder, Value x, Value z) {
  auto k = builder.getCurrentFxpWidth();
  auto f = builder.getCurrentFxpBits();

  // let a = 2^((e+f)/2), that is a[i] = 1 for i = (e+f)/2 else 0
  // let b = lsb(e+f)

  auto z_sep = builder.bitdeintel(z);
  auto lo_mask =
      builder.uint_constant((static_cast<uint128_t>(1) << (k / 2)) - 1);
  auto z_even = builder.and_(z_sep, lo_mask);
  auto z_odd = builder.rshift(z_sep, k / 2);
  z_odd = builder.and_(z_odd, lo_mask);

  // a[i] = z[2*i] ^ z[2*i+1]
  auto a = builder.xor_(z_odd, z_even);
  // b ^= z[2*i]
  auto b = builder.bitparity(z_even, k / 2);

  auto a_rev = builder.bitrev(a, 0, (f / 2) * 2);

  // do compensation
  // Note:
  //   https://arxiv.org/pdf/2107.00501.pdf
  // - the magic number c0 & c1 seems to be wrong.
  // - the LSB algorithm is correct and used in this implementation.
  //
  // The following constant is deduced exactly from:
  //   https://dl.acm.org/doi/10.1145/3411501.3419427
  Value c0;
  Value c1;

  if (f % 2 == 1) {
    c0 = builder.uint_constant(1 << ((f + 3) / 2));                 // f+e even
    c1 = builder.uint_constant((1 << (f / 2 + 1)) * std::sqrt(2));  // f+e odd
  } else {
    c0 = builder.uint_constant((1 << (f / 2)) * std::sqrt(2));  // f+e even
    c1 = builder.uint_constant(1 << (f / 2));                   // f+e odd
  }

  // let comp = 2^(-(e-1)/2) = mux(b, c1, c0) * a_rev
  auto b_vis = builder.getTypeTools().getTypeVisibility(b.getType());
  auto mux_ret = builder.getTypeTools().getType(c0.getType(), b_vis);
  auto mux = builder.select(b, c0, c1, mux_ret);

  auto mul = builder.mul_no_trunc(mux, a_rev);
  return builder.bitcast(mul, x.getType());
}

// Reference:
//  1. https://dl.acm.org/doi/10.1145/3411501.3419427
// Main idea:
//  1. convert x to u * 2^(e + 1) while u belongs to [0.25, 0.5).
//  2. get a nice approximation for u part.
//  3. get the compensation for 2^(e + 1) part.
//  4. multiple two parts and get the result.
Value rsqrt_expand(builder::FxpBuilder &builder, Value input) {
  auto input_int = builder.bitcast(
      input, builder.getIntTypeWithSameWidth(input.getType(), true));

  // let e = NP2(x) , z = 2^(e+f)
  auto z = rsqrt_np2(builder, input_int);

  auto r = rsqrt_init_guess(builder, input, z);
  auto comp = rsqrt_comp(builder, input, z);

  return builder.mul(r, comp);
}

}  // namespace mlir::spu::pphlo::fixedpoint::impl
