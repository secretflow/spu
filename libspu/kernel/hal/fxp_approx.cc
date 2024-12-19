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

#include "libspu/kernel/hal/fxp_approx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>

#include "libspu/core/trace.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/fxp_cleartext.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {
namespace detail {

Value log_minmax_normalized(SPUContext* ctx, const Value& x) {
  static std::array<float, 9> kLogCoefficient{
      0.0,          0.9999964239,  -0.4998741238, 0.3317990258, -0.2407338084,
      0.1676540711, -0.0953293897, 0.0360884937,  -0.0064535442};

  // we have approximation of log(1+x) when x is in [0, 1]
  const auto k1 = constant(ctx, 1.0F, x.dtype(), x.shape());
  auto xm1 = f_sub(ctx, x, k1);

  return detail::polynomial(ctx, xm1, kLogCoefficient, SignType::Positive,
                            SignType::Positive);
}

// Ref:
// Handbook of Mathematical Functions: with Formulas, Graphs, and Mathematical
// Tables, equation 4.1.44
Value log_minmax(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  auto pre_x = _prefix_or(ctx, x);
  const size_t num_fxp_bits = ctx->getFxpBits();

  // because of bitrev, we can only handle x between (0, 2**(fxp+1)),
  // so we can limit the size of bits
  auto k = _popcount(ctx, pre_x, 2 * num_fxp_bits + 2);

  // get most significant non-zero bit of x
  // we avoid direct using detail::highestOneBit for saving one _prefix_or
  auto pre_x1 = _rshift(ctx, pre_x, {1});
  auto msb = _xor(ctx, pre_x, pre_x1);

  // let x = x_norm * factor, where x in [1.0, 2.0)
  auto factor = _bitrev(ctx, msb, 0, 2 * num_fxp_bits + 1).setDtype(x.dtype());
  factor = maskNumberOfBits(ctx, factor, 2 * num_fxp_bits + 1);
  auto norm = f_mul(ctx, x, factor);

  // log(x) = log(x_norm * factor)
  //        = log(x_norm) + log(factor)
  //        = log(x_norm) + (k - fxp_bits - 1) * log(2)
  auto log_norm = log_minmax_normalized(ctx, norm);
  auto log2_e =
      _lshift(ctx, _sub(ctx, k, _constant(ctx, num_fxp_bits + 1, x.shape())),
              {static_cast<int64_t>(num_fxp_bits)})
          .setDtype(x.dtype());
  auto k_log2 = constant(ctx, std::log(2), x.dtype(), x.shape());
  auto log_e = f_mul(ctx, log2_e, k_log2);

  return f_add(ctx, log_norm, log_e);
}

// Pade approximation fo x belongs to [0.5, 1]:
//
// p2524(x) = -0.205466671951 * 10
//          + x * -0.88626599391 * 10
//          + x^2 * 0.610585199015 * 10
//          + x^3 * 0.481147460989 * 10
// q2524(x) = 0.353553425277
//          + x * 0.454517087629 * 10
//          + x^2 * 0.642784209029 * 10
//          + x^3 * 0.1 *10
// log2(x) = p2524(x) / q2524(x)
//
Value log2_pade_normalized(SPUContext* ctx, const Value& x) {
  const auto x2 = f_square(ctx, x);
  const auto x3 = f_mul(ctx, x2, x);

  const auto p0 = constant(ctx, -0.205466671951F * 10, x.dtype(), x.shape());
  const auto p1 = constant(ctx, -0.88626599391F * 10, x.dtype(), x.shape());
  const auto p2 = constant(ctx, 0.610585199015F * 10, x.dtype(), x.shape());
  const auto p3 = constant(ctx, 0.481147460989F * 10, x.dtype(), x.shape());

  const auto q0 = constant(ctx, 0.353553425277F, x.dtype(), x.shape());
  const auto q1 = constant(ctx, 0.454517087629F * 10, x.dtype(), x.shape());
  const auto q2 = constant(ctx, 0.642784209029F * 10, x.dtype(), x.shape());
  const auto q3 = constant(ctx, 0.1F * 10, x.dtype(), x.shape());

  auto p2524 = _mul(ctx, x, p1);
  p2524 = _add(ctx, p2524, _mul(ctx, x2, p2));
  p2524 = _add(ctx, p2524, _mul(ctx, x3, p3));
  p2524 = _add(ctx, _trunc(ctx, p2524), p0).setDtype(x.dtype());

  auto q2524 = _mul(ctx, x, q1);
  q2524 = _add(ctx, q2524, _mul(ctx, x2, q2));
  q2524 = _add(ctx, q2524, _mul(ctx, x3, q3));
  q2524 = _add(ctx, _trunc(ctx, q2524), q0).setDtype(x.dtype());

  return detail::div_goldschmidt(ctx, p2524, q2524);
}

// Refer to
// Chapter 5 Exponentiation and Logarithms
// Benchmarking Privacy Preserving Scientific Operations
// https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf
Value log2_pade(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  const size_t bit_width = SizeOf(ctx->config().field()) * 8;
  auto k = _popcount(ctx, _prefix_or(ctx, x), bit_width);

  const size_t num_fxp_bits = ctx->getFxpBits();

  // let x = x_norm * factor, where x in [0.5, 1.0)
  auto msb = detail::highestOneBit(ctx, x);
  auto factor = _bitrev(ctx, msb, 0, 2 * num_fxp_bits).setDtype(x.dtype());
  factor = maskNumberOfBits(ctx, factor, 2 * num_fxp_bits);
  auto norm = f_mul(ctx, x, factor);

  // log2(x) = log2(x_norm * factor)
  //         = log2(x_norm) + log2(factor)
  //         = log2(x_norm) + (k-fxp_bits)
  return _add(
             ctx, log2_pade_normalized(ctx, norm),
             _lshift(ctx, _sub(ctx, k, _constant(ctx, num_fxp_bits, x.shape())),
                     {static_cast<int64_t>(num_fxp_bits)}))
      .setDtype(x.dtype());
}

// See P11, A.2.4 Logarithm and Exponent,
// https://lvdmaaten.github.io/publications/papers/crypten.pdf
// https://github.com/facebookresearch/CrypTen/blob/master/crypten/common/functions/approximations.py#L55-L104
// Approximates the natural logarithm using 8th order modified
// Householder iterations. This approximation is accurate within 2% relative
// error on [0.0001, 250].
Value log_householder(SPUContext* ctx, const Value& x) {
  Value term_1 = f_div(ctx, x, constant(ctx, 120.0, x.dtype(), x.shape()));
  Value term_2 = f_mul(
      ctx,
      f_exp(ctx,
            f_negate(ctx, f_add(ctx,
                                f_mul(ctx, x,
                                      constant(ctx, 2.0, x.dtype(), x.shape())),
                                constant(ctx, 1.0, x.dtype(), x.shape())))),
      constant(ctx, 20.0, x.dtype(), x.shape()));
  Value y = f_add(ctx, f_sub(ctx, term_1, term_2),
                  constant(ctx, 3.0, x.dtype(), x.shape()));

  const size_t fxp_log_orders = ctx->config().fxp_log_orders();
  SPU_ENFORCE(fxp_log_orders != 0, "fxp_log_orders should not be {}",
              fxp_log_orders);
  std::vector<float> coeffs{0.0};
  for (size_t i = 0; i < fxp_log_orders; i++) {
    coeffs.emplace_back(1.0 / (1.0 + i));
  }

  const size_t num_iters = ctx->config().fxp_log_iters();
  SPU_ENFORCE(num_iters != 0, "fxp_log_iters should not be {}", num_iters);
  for (size_t i = 0; i < num_iters; i++) {
    Value h = f_sub(ctx, constant(ctx, 1.0, x.dtype(), x.shape()),
                    f_mul(ctx, x, f_exp(ctx, f_negate(ctx, y))));
    y = f_sub(ctx, y, detail::polynomial(ctx, h, coeffs));
  }

  return y;
}

// see https://lvdmaaten.github.io/publications/papers/crypten.pdf
//   exp(x) = (1 + x / n) ^ n, when n is infinite large.
Value exp_taylor(SPUContext* ctx, const Value& x) {
  const size_t fxp_exp_iters = ctx->config().fxp_exp_iters();
  SPU_ENFORCE(fxp_exp_iters != 0, "fxp_exp_iters should not be {}",
              fxp_exp_iters);

  Value res = f_add(ctx, _trunc(ctx, x, fxp_exp_iters).setDtype(x.dtype()),
                    constant(ctx, 1.0F, x.dtype(), x.shape()));

  for (size_t i = 0; i < fxp_exp_iters; i++) {
    res = f_square(ctx, res);
  }

  return res;
}

Value exp_prime(SPUContext* ctx, const Value& x) {
  auto clamped_x = x;
  auto offset = ctx->config().experimental_exp_prime_offset();
  auto fxp = ctx->getFxpBits();
  if (!ctx->config().experimental_exp_prime_disable_lower_bound()) {
    // currently the bound is tied to FM128
    SPU_ENFORCE_EQ(ctx->getField(), FieldType::FM128);
    auto lower_bound = (48.0 - offset - 2.0 * fxp) / M_LOG2E;
    clamped_x = _clamp_lower(ctx, clamped_x,
                             constant(ctx, lower_bound, x.dtype(), x.shape()))
                    .setDtype(x.dtype());
  }
  if (ctx->config().experimental_exp_prime_enable_upper_bound()) {
    // currently the bound is tied to FM128
    SPU_ENFORCE_EQ(ctx->getField(), FieldType::FM128);
    auto upper_bound = (124.0 - 2.0 * fxp - offset) / M_LOG2E;
    clamped_x = _clamp_upper(ctx, clamped_x,
                             constant(ctx, upper_bound, x.dtype(), x.shape()))
                    .setDtype(x.dtype());
  }

  auto ret = dynDispatch<spu::Value>(ctx, "exp_a", clamped_x);
  return ret.setDtype(x.dtype());
}

namespace {

// Pade approximation of exp2(x), x is in [0, 1].
// p1015(x) = 0.100000007744302 * 10
//             + x * 0.693147180426163
//             + x^2 * 0.240226510710170
//             + x^3 * 0.555040686204663 / 10
//             + x^4 * 0.961834122588046 / 100
//             + x^5 * 0.133273035928143 / 100
Value exp2_pade_normalized(SPUContext* ctx, const Value& x) {
  auto x2 = f_mul(ctx, x, x);
  auto x3 = f_mul(ctx, x, x2);
  auto x4 = f_mul(ctx, x, x3);
  auto x5 = f_mul(ctx, x, x4);

  const auto p0 = constant(ctx, 0.100000007744302F * 10, x.dtype(), x.shape());
  const auto p1 = constant(ctx, 0.693147180426163F, x.dtype(), x.shape());
  const auto p2 = constant(ctx, 0.240226510710170F, x.dtype(), x.shape());
  const auto p3 = constant(ctx, 0.555040686204663F / 10, x.dtype(), x.shape());
  const auto p4 = constant(ctx, 0.961834122588046F / 100, x.dtype(), x.shape());
  const auto p5 = constant(ctx, 0.133273035928143F / 100, x.dtype(), x.shape());

  auto res = _mul(ctx, x, p1);
  res = _add(ctx, res, _mul(ctx, x2, p2));
  res = _add(ctx, res, _mul(ctx, x3, p3));
  res = _add(ctx, res, _mul(ctx, x4, p4));
  res = _add(ctx, res, _mul(ctx, x5, p5));

  return _add(ctx, _trunc(ctx, res), p0).setDtype(x.dtype());
}

}  // namespace

// Refer to
// Chapter 5 Exponentiation and Logarithms
// Benchmarking Privacy Preserving Scientific Operations
// https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf
// NOTE(junfeng): The valid integer bits of x is 5. Otherwise, the output is
// incorrect.
Value exp2_pade(SPUContext* ctx, const Value& x) {
  const size_t fbits = ctx->getFxpBits();
  const auto k1 = _constant(ctx, 1U, x.shape());
  // TODO(junfeng): Make int_bits configurable.
  const size_t int_bits = 5;
  const size_t bit_width = SizeOf(ctx->getField()) * 8;

  const auto x_bshare = _prefer_b(ctx, x);
  const auto x_msb =
      _rshift(ctx, x_bshare, {static_cast<int64_t>(bit_width - 1)});
  auto x_integer = _rshift(ctx, x_bshare, {static_cast<int64_t>(fbits)});
  auto x_fraction =
      _sub(ctx, x, _lshift(ctx, x_integer, {static_cast<int64_t>(fbits)}))
          .setDtype(x.dtype());
  auto ret = exp2_pade_normalized(ctx, x_fraction);

  for (size_t idx = 0; idx < int_bits; idx++) {
    auto a =
        _and(ctx, _rshift(ctx, x_integer, {static_cast<int64_t>(idx)}), k1);
    a = detail::maskNumberOfBits(ctx, a, 1);
    a = _prefer_a(ctx, a);
    const auto K = 1U << std::min(1UL << idx, bit_width - 2);
    ret = _mul(ctx, ret,
               _add(ctx, _mul(ctx, a, _constant(ctx, K, x.shape())),
                    _sub(ctx, k1, a)))
              .setDtype(ret.dtype());
  }

  // If we could ensure the integer bits of x is 5.
  // we have x, -x, -x_hat. x_hat is 2's complement of -x.
  // Then,
  //             x + (x_hat) = 32
  //            (x_hat) - 32 = -x
  //  exp2(x_hat) / exp2(32) = exp(-x)
  //  so exp(-x) = exp2(x_hat) / exp2(32)
  auto ret_reciprocal =
      _trunc(ctx, ret, std::pow(2, int_bits)).setDtype(ret.dtype());

  // ret + msb * (reciprocal - ret)
  return f_add(
      ctx, ret,
      _mul(ctx, x_msb, f_sub(ctx, ret_reciprocal, ret)).setDtype(ret.dtype()));
}

Value exp_pade(SPUContext* ctx, const Value& x) {
  return f_exp2(ctx, f_mul(ctx, x,
                           constant(ctx, std::log2(std::exp(1.0F)), x.dtype(),
                                    x.shape())));
}

// Refer to
// https://www.wolframalpha.com/input?i=Pade+approximation+tanh%28x%29+order+5%2C5.
// tanh(x) = (x + x^3 / 9.0 + x^5 /945.0) /
//           (1 + 4 * x^2 / 9.0 + x^4 / 63.0)
Value tanh_pade(SPUContext* ctx, const Value& x) {
  const auto x_2 = f_square(ctx, x);
  const auto x_4 = f_square(ctx, x_2);

  // Idea here...
  // transform formula into
  // x * (1 + x^2  / 9 + x^4 / 945) / (1 + 4 * x^2 / 9 + x^4 / 63)
  // = x * (945 + 105 * x^2 + x^4) / (945 + 420 * x^2 + 15 * x^4)
  // This can save some truncations

  const auto c_945 = constant(ctx, 945.0F, x.dtype(), x.shape());
  const auto c_105 = constant(ctx, 105, DT_I32, x.shape());
  const auto c_420 = constant(ctx, 420, DT_I32, x.shape());
  const auto c_15 = constant(ctx, 15, DT_I32, x.shape());

  const auto x_2_m_105 = _mul(ctx, x_2, c_105).setDtype(x_2.dtype());
  const auto x_2_m_420 = _mul(ctx, x_2, c_420).setDtype(x_2.dtype());
  const auto x_4_m_15 = _mul(ctx, x_4, c_15).setDtype(x_4.dtype());

  const auto nominator =
      f_mul(ctx, x, f_add(ctx, c_945, f_add(ctx, x_2_m_105, x_4)));

  const auto denominator = f_add(ctx, c_945, f_add(ctx, x_2_m_420, x_4_m_15));

  return f_div(ctx, nominator, denominator);
}

// Reference:
// https://github.com/facebookresearch/CrypTen/blob/6ef151101668591bcfb2bbf7e7ebd39ab6db0413/crypten/common/functions/approximations.py#L365
Value compute_chebyshev_polynomials(SPUContext* ctx, const Value& x,
                                    int64_t terms) {
  // Ref:
  // https://en.wikipedia.org/wiki/Chebyshev_polynomials#Recurrence_definition
  // Chebyshev Polynomials of the first kind are defined as
  //.. math::
  //    P_0(x) = 1
  //    P_1(x) = x
  //    P_{n+1}(x) = 2xP_{n}(x) - P_{n-1}(x)
  std::vector<Value> poly = {x};

  // y = 4*x^2 - 2
  auto four = constant(ctx, 4, DT_I32, x.shape());
  auto two = constant(ctx, 2.0F, x.dtype(), x.shape());
  auto y =
      f_sub(ctx, _mul(ctx, four, f_square(ctx, x)).setDtype(x.dtype()), two);
  // z = y - 1
  auto one = constant(ctx, 1.0F, x.dtype(), x.shape());
  auto z = f_sub(ctx, y, one);

  poly.emplace_back(f_mul(ctx, x, z));

  for (int64_t idx = 2; idx < terms; ++idx) {
    // next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
    auto next = f_sub(ctx, f_mul(ctx, y, poly[idx - 1]), poly[idx - 2]);
    poly.emplace_back(std::move(next));
  }

  return concatenate(ctx, poly, 0);
}

Value tanh_chebyshev(SPUContext* ctx, const Value& x) {
  // Cheb coeff, deg = 17, domain = [-5,5]
  static const std::array<float, 9> kCoeffs = {
      1.2514045938932097,   -0.3655987797163166,   0.17253141478140663,
      -0.08943445792774211, 0.047703017901250824,  -0.025830290571688078,
      0.014338801903468182, -0.008541730970059077, 0.0061230685785789475};

  auto coeff_value = constant(ctx, kCoeffs, x.dtype(),
                              {1, static_cast<int64_t>(kCoeffs.size())});

  auto normalized_x = reshape(ctx, x, {1, x.numel()});

  normalized_x =
      _clamp(ctx, normalized_x,
             constant(ctx, -5.0F, normalized_x.dtype(), normalized_x.shape()),
             constant(ctx, 5.0F, normalized_x.dtype(), normalized_x.shape()))
          .setDtype(x.dtype());

  normalized_x = f_mul(
      ctx, constant(ctx, 0.2F, x.dtype(), normalized_x.shape()), normalized_x);
  auto poly = compute_chebyshev_polynomials(ctx, normalized_x, kCoeffs.size());

  auto ret = f_mmul(ctx, coeff_value, poly);

  return reshape(ctx, ret, x.shape());
}

Value sin_chebyshev(SPUContext* ctx, const Value& x) {
  // Cheb coeff, deg = 9, domain = [-1.25*pi, 1.25*pi]
  // use larger domain for accurate output on boundary
  static const std::array<float, 5> kCoeffs = {
      -0.07570787578233389, -0.8532364056408055, 0.2474789050491474,
      -0.02719844932262742, 0.0016750058127101841};

  auto coeff_value = constant(ctx, kCoeffs, x.dtype(),
                              {1, static_cast<int64_t>(kCoeffs.size())});

  // Normalize input to[-pi, pi]
  // theta - TWO_PI * Math.floor((theta + Math.PI) / TWO_PI)
  auto pi = constant(ctx, M_PI, x.dtype(), x.shape());
  auto two_pi = constant(ctx, 2 * M_PI, x.dtype(), x.shape());
  auto two_pi_inv = constant(ctx, 1 / (2 * M_PI), x.dtype(), x.shape());
  auto normalized = f_mul(ctx, f_add(ctx, x, pi), two_pi_inv);
  normalized = f_mul(ctx, f_floor(ctx, normalized), two_pi);
  normalized = f_sub(ctx, x, normalized);

  normalized = reshape(ctx, normalized, {1, normalized.numel()});

  // rescale the original x
  normalized = f_mul(ctx,
                     constant(ctx, 0.25464790894703254F, normalized.dtype(),
                              normalized.shape()),
                     normalized);

  auto poly = compute_chebyshev_polynomials(ctx, normalized, kCoeffs.size());

  auto ret = f_mmul(ctx, coeff_value, poly);

  return reshape(ctx, ret, x.shape());
}

Value cos_chebyshev(SPUContext* ctx, const Value& x) {
  auto half_pi = constant(ctx, M_PI / 2, x.dtype(), x.shape());
  // cos(x) = sin(pi/2 - x)
  return sin_chebyshev(ctx, f_sub(ctx, half_pi, x));
}

}  // namespace detail

Value f_exp(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_exp_p(ctx, x);
  }

  switch (ctx->config().fxp_exp_mode()) {
    case RuntimeConfig::EXP_DEFAULT:
    case RuntimeConfig::EXP_TAYLOR:
      return detail::exp_taylor(ctx, x);
    case RuntimeConfig::EXP_PADE: {
      // The valid input for exp_pade is [-kInputLimit, kInputLimit].
      // TODO(junfeng): should merge clamp into exp_pade to save msb ops.
      const float kInputLimit = 32.0 / std::log2(std::exp(1));
      const auto clamped_x =
          _clamp(ctx, x, constant(ctx, -kInputLimit, x.dtype(), x.shape()),
                 constant(ctx, kInputLimit, x.dtype(), x.shape()))
              .setDtype(x.dtype());
      return detail::exp_pade(ctx, clamped_x);
    }
    case RuntimeConfig::EXP_PRIME:
      if (ctx->hasKernel("exp_a")) {
        return detail::exp_prime(ctx, x);
      } else {
        SPU_THROW(
            "exp_a is not implemented for this protocol, currently only "
            "2pc "
            "semi2k is supported.");
      }
    default:
      SPU_THROW("unexpected exp approximation method {}",
                ctx->config().fxp_exp_mode());
  }
}

Value f_log(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_log_p(ctx, x);
  }

  switch (ctx->config().fxp_log_mode()) {
    // Note:
    // Generally, MINMAX approximation is a fast and precise DEFAULT option
    // which gives very high precision (avg error < 3e-5) when x is between ( 0,
    // 2**(fxp_bits + 1) ), and you may not need to change it.
    // There are still some cases that householder approximation is faster:
    //    - 1. SEMI2K is used and N (total parties) is very large (maybe > 10).
    //    - 2. CHEETAH is used.
    // The reason for this is that the implementation of householder
    // approximation does not involve range reduction (`a2b` and `b2a` for MPC
    // protocols).
    // However, the valid input range for householder approximation is very
    // limited, so you MUST ensure that x is not very close to zero,
    // nor too large before using it.
    case RuntimeConfig::LOG_DEFAULT:
    case RuntimeConfig::LOG_MINMAX:
      return detail::log_minmax(ctx, x);
    case RuntimeConfig::LOG_PADE:
      return f_mul(ctx, constant(ctx, std::log(2.0F), x.dtype(), x.shape()),
                   f_log2(ctx, x));
    case RuntimeConfig::LOG_NEWTON:
      return detail::log_householder(ctx, x);
    default:
      SPU_THROW("unexpected log approximation method {}",
                ctx->config().fxp_log_mode());
  }
}

Value f_log1p(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_log(ctx, f_add(ctx, constant(ctx, 1.0F, x.dtype(), x.shape()), x));
}

Value f_log2(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return detail::log2_pade(ctx, x).setDtype(x.dtype());
}

Value f_exp2(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  return detail::exp2_pade(ctx, x);
}

Value f_tanh(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

#ifndef TANH_USE_PADE
  return detail::tanh_chebyshev(ctx, x);
#elif
  // For tanh inputs beyond [-3, 3], result is infinitely close to -1, 1
  // pade approximation has a relative ok result between [-3, 3], so clamp
  // inputs to this range.
  auto normalized_x = _clamp(ctx, x, constant(ctx, -3.F, x.dtype(), x.shape()),
                             constant(ctx, 3.F, x.dtype(), x.shape()))
                          .setDtype(x.dtype());

  return detail::tanh_pade(ctx, normalized_x);
#endif
}

static Value rsqrt_init_guess(SPUContext* ctx, const Value& x, const Value& z) {
  SPU_TRACE_HAL_LEAF(ctx, x, z);
  const size_t f = ctx->getFxpBits();

  // let u in [0.25, 0.5)
  auto z_rev = _bitrev(ctx, z, 0, 2 * f);
  z_rev = detail::maskNumberOfBits(ctx, z_rev, 2 * f);

  auto u = _trunc(ctx, _mul(ctx, x, z_rev)).setDtype(x.dtype());

  // let rsqrt(u) = 26.02942339 * u^4 - 49.86605845 * u^3 + 38.4714796 * u^2
  // - 15.47994394 * u + 4.14285016
  spu::Value r;
  if (!ctx->config().enable_lower_accuracy_rsqrt()) {
    auto coeffs = {0.0F, -15.47994394F, 38.4714796F, -49.86605845F,
                   26.02942339F};
    r = f_add(ctx,
              detail::polynomial(ctx, u, coeffs, SignType::Positive,
                                 SignType::Positive),
              constant(ctx, 4.14285016F, x.dtype(), x.shape()));
  } else {
    auto coeffs = {0.0F, -5.9417F, 4.7979F};
    r = f_add(ctx,
              detail::polynomial(ctx, u, coeffs, SignType::Positive,
                                 SignType::Positive),
              constant(ctx, 3.1855F, x.dtype(), x.shape()));
  }

  return r;
}

static Value rsqrt_comp(SPUContext* ctx, const Value& x, const Value& z) {
  SPU_TRACE_HAL_LEAF(ctx, x, z);

  const size_t k = SizeOf(ctx->getField()) * 8;
  const size_t f = ctx->getFxpBits();

  // let a = 2^((e+f)/2), that is a[i] = 1 for i = (e+f)/2 else 0
  // let b = lsb(e+f)
  Value a;
  Value b;
  {
    auto z_sep = _bitdeintl(ctx, z);
    auto lo_mask =
        _constant(ctx, (static_cast<uint128_t>(1) << (k / 2)) - 1, x.shape());
    auto z_even = _and(ctx, z_sep, lo_mask);
    auto z_odd =
        _and(ctx, _rshift(ctx, z_sep, {static_cast<int64_t>(k / 2)}), lo_mask);

    // a[i] = z[2*i] ^ z[2*i+1]
    a = _xor(ctx, z_odd, z_even);
    // b ^= z[2*i]
    b = _bit_parity(ctx, z_even, k / 2);
  }

  auto a_rev = _bitrev(ctx, a, 0, (f / 2) * 2);
  a_rev = detail::maskNumberOfBits(ctx, a_rev, (f / 2) * 2);

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
    c0 = _constant(ctx, 1 << ((f + 3) / 2), x.shape());  // f+e even
    c1 = _constant(ctx, (1 << (f / 2 + 1)) * std::sqrt(2),
                   x.shape());  // f+e odd
  } else {
    c0 = _constant(ctx, (1 << (f / 2)) * std::sqrt(2),
                   x.shape());                     // f+e even
    c1 = _constant(ctx, 1 << (f / 2), x.shape());  // f+e odd
  }

  // let comp = 2^(-(e-1)/2) = mux(b, c1, c0) * a_rev
  return _mul(ctx, _mux(ctx, b, c0, c1), a_rev);
}

static Value rsqrt_np2(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // let e = NP2(x), z = 2^(e+f)
  return _lshift(ctx, detail::highestOneBit(ctx, x), {1});
}

// Reference:
//  1. https://dl.acm.org/doi/10.1145/3411501.3419427
// Main idea:
//  1. convert x to u * 2^(e + 1) while u belongs to [0.25, 0.5).
//  2. get a nice approximation for u part.
//  3. get the compensation for 2^(e + 1) part.
//  4. multiple two parts and get the result.
Value f_rsqrt(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // let e = NP2(x) , z = 2^(e+f)
  auto z = rsqrt_np2(ctx, x);

  // TODO: we should avoid fork context in hal layer, it will make global
  // scheduling harder and also make profiling harder.
  if (ctx->config().experimental_enable_intra_op_par()) {
    auto sub_ctx = ctx->fork();
    auto r = std::async(rsqrt_init_guess,
                        dynamic_cast<SPUContext*>(sub_ctx.get()), x, z);
    auto comp = rsqrt_comp(ctx, x, z);
    return _trunc(ctx, _mul(ctx, r.get(), comp)).setDtype(x.dtype());
  } else {
    auto r = rsqrt_init_guess(ctx, x, z);
    auto comp = rsqrt_comp(ctx, x, z);
    return _trunc(ctx, _mul(ctx, r, comp)).setDtype(x.dtype());
  }
}

// Reference:
// 1. https://eprint.iacr.org/2012/405.pdf, section 6.1
// 2.
// https://github.com/tf-encrypted/tf-encrypted/blob/3b0f14d26e900caf12a92a9ea2284ccd4d58e683/tf_encrypted/protocol/aby3/fp.py#L35-L52
// Goldschmidt iteration, needs an initial approximation of sqrt_inv(x).
// In the end, g is an approximation of sqrt(x) while h is an approximation of
// 1 / (2 * sqrt(x)).
Value f_sqrt(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  const auto c0 = constant(ctx, 0.5F, x.dtype(), x.shape());
  const auto c1 = constant(ctx, 1.5F, x.dtype(), x.shape());

  Value y0 = f_rsqrt(ctx, x);
  Value g = f_mul(ctx, x, y0);
  Value h = f_mul(ctx, y0, c0);

  // iterations of 1 is enough.
  const int iterations = 1;

  for (int i = 0; i < iterations; i++) {
    const auto r = f_sub(ctx, c1, f_mul(ctx, g, h));
    g = f_mul(ctx, g, r);
    h = f_mul(ctx, h, r);
  }

  return g;
}

namespace {

Value sigmoid_real(SPUContext* ctx, const Value& x) {
  // f(x) = 1/(1+exp(-x))
  const auto c1 = constant(ctx, 1.0F, x.dtype(), x.shape());
  return f_reciprocal(ctx, f_add(ctx, c1, f_exp(ctx, f_negate(ctx, x))));
}

Value sigmoid_mm1(SPUContext* ctx, const Value& x) {
  // SigmoidMM1: f(x) = 0.5 + 0.125 * x
  const auto c1 = constant(ctx, 0.5F, x.dtype(), x.shape());
  const auto c2 = constant(ctx, 0.125F, x.dtype(), x.shape());
  return f_add(ctx, c1, f_mul(ctx, c2, x));
}

Value sigmoid_seg3(SPUContext* ctx, const Value& x) {
  // f(x) = 0.5 + 0.125x if -4 <= x <= 4
  //        1            if       x > 4
  //        0            if  -4 > x
  // Rounds = Gt + Mux*2 = 4 + Log(K)
  auto upper = constant(ctx, 1.0F, x.dtype(), x.shape());
  auto lower = constant(ctx, 0.0F, x.dtype(), x.shape());
  auto middle = sigmoid_mm1(ctx, x);

  auto upper_bound = constant(ctx, 4.0F, x.dtype(), x.shape());
  auto lower_bound = constant(ctx, -4.0F, x.dtype(), x.shape());

  auto ret = _mux(ctx, f_less(ctx, upper_bound, x), upper, middle);
  return _mux(ctx, f_less(ctx, x, lower_bound), lower, ret).setDtype(x.dtype());
}

}  // namespace

Value f_sigmoid(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  switch (ctx->config().sigmoid_mode()) {
    case RuntimeConfig::SIGMOID_DEFAULT:
    case RuntimeConfig::SIGMOID_MM1: {
      return sigmoid_mm1(ctx, x);
    }
    case RuntimeConfig::SIGMOID_SEG3: {
      return sigmoid_seg3(ctx, x);
    }
    case RuntimeConfig::SIGMOID_REAL: {
      return sigmoid_real(ctx, x);
    }
    default: {
      SPU_THROW("Should not hit");
    }
  }
}

Value f_sine(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_sine_p(ctx, x);
  }

  return detail::sin_chebyshev(ctx, x);
}

Value f_cosine(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_cosine_p(ctx, x);
  }

  return detail::cos_chebyshev(ctx, x);
}

namespace {

Value ErfImpl(SPUContext* ctx, const Value& x) {
  static std::array<float, 5> kErfCoefficient{1.0, 0.278393, 0.230389, 0.000972,
                                              0.078108};
  auto one = constant(ctx, 1.0, x.dtype(), x.shape());

  auto z = detail::polynomial(ctx, x, kErfCoefficient, SignType::Positive,
                              SignType::Positive);
  z = f_square(ctx, z);
  z = f_square(ctx, z);
  z = detail::reciprocal_goldschmidt_positive(ctx, z);

  return f_sub(ctx, one, z);
}

}  // namespace

// Ref:
// Handbook of Mathematical Functions: with Formulas, Graphs, and Mathematical
// Tables, equation 7.1.27, maximum absolute error <= 5e-4
Value f_erf(SPUContext* ctx, const Value& x) {
  if (x.isPublic()) {
    return f_erf_p(ctx, x);
  }
  auto zero = constant(ctx, 0.0, x.dtype(), x.shape());
  auto pred = f_less(ctx, x, zero);

  auto abs_x = f_abs(ctx, x);

  auto three = constant(ctx, 3.0, x.dtype(), x.shape());
  auto cond = f_less(ctx, abs_x, three);

  auto erf = ErfImpl(ctx, abs_x);

  // we do this truncation because:
  // 1. for large abs_x, reciprocal may overflow
  // 2. error is sufficiently small (< 2.2e-5)
  erf = _mux(ctx, cond, erf, constant(ctx, 1.0F, x.dtype(), x.shape()))
            .setDtype(x.dtype());

  return _mux(ctx, pred, f_negate(ctx, erf), erf).setDtype(x.dtype());
}

namespace {
Value AtanApproxLocal(SPUContext* ctx, const Value& x) {
  // 6-order minimax approximation with max error < 6.3893490851163973e-6
  static std::array<float, 7> kAtanCoefficientSmall{
      6.3893490851163976e-06, 0.99938232039482577, 0.0096717091887422429,
      -0.38851091678439126,   0.13850820695354954, 0.065822467870128534,
      -0.039488402923576769};

  // 10-order minimax approximation with max error < 1.4802815832055511e-9
  static std::array<float, 11> kAtanCoefficientLarge{
      7.3035884235708622e-09, 0.99999906394905635,   3.5324890092487464e-05,
      -0.33393042345794194,   0.0054765660426422556, 0.16982068444578205,
      0.10531189733914688,    -0.37905943050720364,  0.32946653597875702,
      -0.1337452245060563,    0.022023163399866309};

  if (ctx->getFxpBits() <= 20) {
    return detail::polynomial(ctx, x, kAtanCoefficientSmall, SignType::Positive,
                              SignType::Positive);
  } else {
    return detail::polynomial(ctx, x, kAtanCoefficientLarge, SignType::Positive,
                              SignType::Positive);
  }
}

Value atan2_minimax(SPUContext* ctx, const Value& y, const Value& x) {
  auto common_type = _common_type(ctx, x.storage_type(), y.storage_type());
  std::vector<Value> xy = {_cast_type(ctx, x, common_type).setDtype(x.dtype()),
                           _cast_type(ctx, y, common_type).setDtype(y.dtype())};
  // vectorize the computation of the msb and abs of x and y
  std::vector<Value> msb_xy;
  spu::vmap(xy.begin(), xy.end(), std::back_inserter(msb_xy),
            [&](const Value& v) { return _prefer_a(ctx, _msb(ctx, v)); });
  std::vector<Value> abs_xy;
  spu::vmap(msb_xy.begin(), msb_xy.end(), xy.begin(), xy.end(),
            std::back_inserter(abs_xy), [&](const Value& xx, const Value& yy) {
              return _mux(ctx, xx, _negate(ctx, yy), yy);
            });

  auto cmp = _less(ctx, abs_xy[0], abs_xy[1]);
  cmp = _prefer_a(ctx, cmp);

  auto bigger = _mux(ctx, cmp, abs_xy[1], abs_xy[0]).setDtype(x.dtype());
  auto smaller =
      _sub(ctx, _add(ctx, abs_xy[0], abs_xy[1]), bigger).setDtype(x.dtype());

  // we fix tan(\theta) in [0,1] here for better acc/perf both, and do the
  // re-mapping to full circle in the end.
  auto tangent = detail::div_goldschmidt_general(
      ctx, smaller, bigger, SignType::Positive, SignType::Positive);

  // approximation of arctan(tangent) when tangent is in [0,1]
  auto theta = AtanApproxLocal(ctx, tangent);

  // To do re-mapping:
  //   1. if abs_y > abs_x (indeed, we compute cot(\theta) before), then \theta
  //   = pi/2 - \theta
  //   2. if x < 0 (we compute tan(pi - \theta) before), then \theta = pi -
  //   \theta
  //   3. if y < 0 (we compute tan(-\theta) before), then \theta = -\theta
  theta = _mux(ctx, cmp,
               _sub(ctx, constant(ctx, M_PI_2, x.dtype(), x.shape()), theta),
               theta);
  theta =
      _mux(ctx, msb_xy[0],
           _sub(ctx, constant(ctx, M_PI, x.dtype(), x.shape()), theta), theta);

  theta = _mux(ctx, msb_xy[1], _negate(ctx, theta), theta).setDtype(x.dtype());

  return theta;
}
}  // namespace

Value f_atan2(SPUContext* ctx, const Value& y, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, y, x);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype() &&
              y.shape() == x.shape());

  if (x.isPublic() && y.isPublic()) {
    return f_atan2_p(ctx, y, x);
  }

  return atan2_minimax(ctx, y, x);
}

namespace {

// ref: Handbook of Mathematical Functions: with Formulas, Graphs, and
// Mathematical
Value acos_minimax(SPUContext* ctx, const Value& x) {
  auto msb = _msb(ctx, x);
  msb = _prefer_a(ctx, msb);

  auto abs_x = _mux(ctx, msb, _negate(ctx, x), x).setDtype(x.dtype());

  // arccos(x) ~= sqrt(1-x) * poly(x), when x is in [0,1]
  // 3-order minimax approximation with max error < 5e-5
  static std::array<float, 4> kAcosCoefficientSmall{1.5707288, -0.2121144,
                                                    0.0742610, -0.0187293};

  // 7-order minimax approximation with max error < 2e-8
  static std::array<float, 8> kAcosCoefficientLarge{
      1.5707963050, -0.2145988016, 0.0889789874, -0.0501743046,
      0.0308918810, -0.0170881256, 0.0066700901, -0.0012624911};

  Value poly_part;
  if (ctx->getFxpBits() <= 20) {
    poly_part = detail::polynomial(ctx, abs_x, kAcosCoefficientSmall);
  } else {
    poly_part = detail::polynomial(ctx, abs_x, kAcosCoefficientLarge);
  }
  const auto k1 = constant(ctx, 1.0F, x.dtype(), x.shape());
  auto sqrt_part = f_sqrt(ctx, f_sub(ctx, k1, abs_x));

  auto ret = f_mul(ctx, sqrt_part, poly_part, SignType::Positive);

  const auto pi = constant(ctx, M_PI, x.dtype(), x.shape());
  ret = _mux(ctx, msb, f_sub(ctx, pi, ret), ret).setDtype(x.dtype());

  return ret;
}

}  // namespace

Value f_acos(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_acos_p(ctx, x);
  }

  return acos_minimax(ctx, x);
}

Value f_asin(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_asin_p(ctx, x);
  }

  const auto k_pi2 = constant(ctx, M_PI_2, x.dtype(), x.shape());
  // asin(x) = pi/2 - acos(x)
  return f_sub(ctx, k_pi2, f_acos(ctx, x));
}
}  // namespace spu::kernel::hal
