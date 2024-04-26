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

#include "experimental/squirrel/objectives.h"

#include "experimental/squirrel/utils.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_ternary.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/shift.h"
#include "libspu/mpc/cheetah//type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace squirrel {

// We compute the alternative Gain = |G|*rsqrt(H + lambda)
// This is more numerically stable than G^2/(H + lambda) since G^2 might be huge
// and overflow the 2^k ring.
// TODO(lwj) To test the performance G * (G / (H + lambda))
static spu::Value ComputeGain(spu::SPUContext* ctx, const spu::Value& G,
                              const spu::Value& H, double lambda) {
  namespace skh = spu::kernel::hlo;

  SPU_ENFORCE(G.shape() == H.shape());
  SPU_ENFORCE(G.isFxp() and H.isFxp());
  auto abs_G = skh::Abs(ctx, G);
  // NOTE(lwj): can use iteration=1 to improve the precision of rsqrt
  auto rsqrt_H =
      Rsqrt(ctx, skh::Add(ctx, H, skh::Constant(ctx, lambda, H.shape())),
            /*iteration*/ 0);
  // NOTE(lwj): Rsqrt function can handle a larger range than hlo::Rsqrt
  // clang-format off
  // auto rsqrt_H = skh::Rsqrt(ctx, skh::Add(ctx, H, skh::Constant(ctx, lambda, H.shape())));
  // clang-format on
  return skh::Mul(ctx, abs_G, rsqrt_H);
}

// y = c0 + x*c1 + x^2*c2 + x^3*c3
static spu::Value polynomial_3(spu::SPUContext* ctx, const spu::Value& x,
                               absl::Span<spu::Value const> coeffs);

static spu::Value polynomial_3(spu::SPUContext* ctx, const spu::Value& x,
                               absl::Span<float const> coeffs) {
  SPU_ENFORCE_EQ(coeffs.size(), 4UL);
  std::vector<spu::Value> cs;
  cs.reserve(coeffs.size());
  for (const auto& c : coeffs) {
    cs.push_back(spu::kernel::hal::constant(ctx, c, x.dtype(), x.shape()));
  }
  return polynomial_3(ctx, x, cs);
}

// NOTE(lwj): basically the same in `libspu/kernel/hal/fxp_approx.cc`
// However here we can handle a wider input range `x`.
// That is 2^{-fxp} <= x < 2^{2*fxp}.
// As the cost, we need to set the fixed-point precision fxp such that
// 3 * fxp + 2 < k for the 2^k ring.
spu::Value Rsqrt(spu::SPUContext* ctx, const spu::Value& x, int iterations) {
  using namespace spu::kernel;

  const size_t k = SizeOf(ctx->getField()) * 8;
  const size_t f = ctx->getFxpBits();
  if (3 * f + 2 >= k) {
    // switch back to spu's implementation
    return spu::kernel::hlo::Rsqrt(ctx, x);
  }

  auto mul_positive = [&](const spu::Value& x, const spu::Value& y,
                          size_t bit) {
    return hal::_trunc(ctx, hal::_mul(ctx, x, y), bit, spu::SignType::Positive)
        .setDtype(x.dtype());
  };

  // x = c * 2^{m + f} for c \in [0.5, 1.)
  // z = 2^{m + f - 1} i.e z = 2^{floor(log2(x))}
  auto z = hal::detail::highestOneBit(ctx, x);

  // z' = 2^{2f - m}
  auto zhat = hal::_bitrev(ctx, z, 0, 3 * f);
  hal::detail::hintNumberOfBits(zhat, 3 * f);

  // x * z = c * 2^{3f} -- trunc 2f --> c * 2^{f}
  // c \in [0.5, 1.0) with 2^f precision
  auto c = mul_positive(x, zhat, 2 * f);
  auto c2 = hal::f_square(ctx, c);

  // r \approx rsqrt(c) given c \in [0.5, 1.0)
  auto r = hal::f_add(
      ctx, hlo::Constant(ctx, 2.22391271, x.shape()),
      hal::_trunc(
          ctx,
          hal::_add(
              ctx,
              hal::_mul(ctx, c, hlo::Constant(ctx, -2.04764217, x.shape())),
              hal::_mul(ctx, c2, hlo::Constant(ctx, 0.82868548, x.shape()))))
          .setDtype(x.dtype()));

  // x = c * 2^{f + m} for c \in [0.5, 1)
  // r \approx inv_sqrt(0.5*c)
  // z' = 2^{2f - m}
  // a = 2^{(2f - m) // 2}
  // b = is_even(2f - m)
  spu::Value a;
  spu::Value b;
  {
    auto z_sep = hal::_bitdeintl(ctx, zhat);
    auto lo_mask = hal::_constant(
        ctx, (static_cast<uint128_t>(1) << (k / 2)) - 1, x.shape());
    auto z_even = hal::_and(ctx, z_sep, lo_mask);
    auto z_odd = hal::_and(ctx, hal::_rshift(ctx, z_sep, k / 2), lo_mask);

    // a[i] = z[2*i] ^ z[2*i+1]
    a = hal::_xor(ctx, z_odd, z_even);
    // b ^= z[2*i]
    b = hal::_bit_parity(ctx, z_even, k / 2).setDtype(spu::DT_I1);
    hal::detail::hintNumberOfBits(b, 1);
  }

  // x = c * 2^{m}
  // r = rsqrt(c)
  //
  // rsqrt(x) = r * 2^{-m/2}
  //
  // a = 2^{(2f - m)//2}
  // b = is_even(2f - m)
  // b = 1 -> a' <- a
  // b = 0 -> a' <- a * sqrt(2)
  // a' = 2^{f - m/2}
  auto even_choice =
      hlo::Constant(ctx, static_cast<int64_t>((1LL << f)), x.shape());
  auto odd_choice = hlo::Constant(
      ctx, static_cast<int64_t>((1L << f) * std::sqrt(2.0)), x.shape());

  a = mul_positive(a, hlo::Select(ctx, b, even_choice, odd_choice), f);
  auto inv_sqrt_init = mul_positive(r, a, f);

  if (iterations <= 0) {
    return inv_sqrt_init;
  }

  const auto c0 = hlo::Constant(ctx, 0.5, x.shape());
  const auto c1 = hlo::Constant(ctx, 1.5, x.shape());

  // g \approx sqrt(x)
  // h \approx 0.5*rsqrt(x)
  auto g = mul_positive(x, inv_sqrt_init, f);
  auto h = mul_positive(c0, inv_sqrt_init, f);

  // Goldschmidt iteration,
  // g -> sqrt(x)
  // h -> 0.5*rsqrt(x))
  for (int i = 0; i < iterations; i++) {
    auto r = mul_positive(g, h, f);
    r = hal::f_sub(ctx, c1, r);
    if (i + 1 < iterations) {
      g = mul_positive(g, r, f);
    }
    h = mul_positive(h, r, f);
  }

  return hlo::Lshift(ctx, h, hlo::Constant(ctx, 1, h.shape()));
}

// clang-format off
// Gs, Hs: shape (n, m)
//   Gs[i, j] is the sum of the first j bins, ie., Gs[i, j] = sum_{k <= j} bin_{i, k}
// Thus the last bucket indicate the gradient sum of all samples, ie, Ga and Ha.
// A split at the j-th bucket is given by `Gl = Gs[:, j]` and `Gr = Gs[:, -1] - Gl`.
// The corresponding gain is
//   gain[i, j] = Gl^2/(Hl + reg) + Gr^2/(Hr + reg) - Ga^2/(Ha + reg)
// Find the best split with a maximum gain, ie., ArgMax_j gain[:, j]
//
// NOTE: we use |Gl|*Rsqrt(Hl + reg) as the alternative objective function
// which is more numerically stable than using the square version Gl^2/(Hl + reg).
// clang-format on
spu::Value MaxGainOnLevel(spu::SPUContext* ctx, const spu::Value& Gs,
                          const spu::Value& Hs, double reg_lambda) {
  // REF
  // `secretflow/ml/boost/ss_xgb_v/core/node_split.py#find_best_split_bucket`
  namespace skh = spu::kernel::hlo;

  constexpr int64_t kDim = 2;
  SPU_ENFORCE_EQ(Gs.shape(), Hs.shape());
  SPU_ENFORCE_EQ(Gs.shape().ndim(), kDim);
  SPU_ENFORCE(Gs.numel() > 1);

  spu::Index ends{Gs.shape()};
  spu::Index starts = {0, ends.back() - 1};
  spu::Strides strides(kDim, 1);

  // last buckets is the total gradient sum of all samples belong to current
  // level nodes.
  auto GA = skh::Broadcast(ctx, skh::Slice(ctx, Gs, starts, ends, strides),
                           Gs.shape(), {});
  auto HA = skh::Broadcast(ctx, skh::Slice(ctx, Hs, starts, ends, strides),
                           Hs.shape(), {});

  // gradient sums of left child nodes after splitting by each bucket
  const auto& GL = Gs;
  const auto& HL = Hs;
  // gradient sums of right child nodes after splitting by each bucket
  auto GR = skh::Sub(ctx, GA, GL);
  auto HR = skh::Sub(ctx, HA, HL);

  auto tmp_G = skh::Concatenate(ctx, {GL, GR}, 0);
  auto tmp_H = skh::Concatenate(ctx, {HL, HR}, 0);
  auto gain = ComputeGain(ctx, tmp_G, tmp_H, reg_lambda);

  auto gainL = skh::Slice(ctx, gain, {0, 0}, spu::Index{GL.shape()}, {1, 1});
  auto gainR = skh::Slice(ctx, gain, {GL.shape()[0], 0},
                          spu::Index{gain.shape()}, {1, 1});

  // last objective value means split all sample to left, equal to no split.
  auto gainAll = skh::Broadcast(
      ctx, skh::Slice(ctx, gainL, starts, ends, strides), gainL.shape(), {});

  // ArgMax(gainL + gainR - gainAll)
  auto _gain = skh::Sub(ctx, skh::Add(ctx, gainL, gainR), gainAll);
  return ArgMax(ctx, _gain, /*axis*/ 1);
}

namespace {

[[maybe_unused]] spu::NdArrayRef CastRing(const spu::NdArrayRef& in,
                                          const spu::FieldType& ftype) {
  using namespace spu;
  const auto field = in.eltype().as<RingTy>()->field();
  const auto numel = in.numel();
  const size_t k = SizeOf(field) * 8;
  const size_t to_bits = SizeOf(ftype) * 8;

  if (to_bits == k) {
    // euqal ring size, do nothing
    return in;
  }

  NdArrayRef res;
  if (in.eltype().isa<mpc::cheetah::BShrTy>()) {
    const auto* in_type = in.eltype().as<mpc::cheetah::BShrTy>();
    SPU_ENFORCE(in_type->nbits() <= to_bits);
    res = NdArrayRef(makeType<mpc::cheetah::BShrTy>(ftype, in_type->nbits()),
                     in.shape());
  } else {
    // For AShr, support cast down only
    SPU_ENFORCE(to_bits <= k, "src_bits= {}, to_bits={}", k, to_bits);
    res = NdArrayRef(makeType<mpc::cheetah::AShrTy>(ftype), in.shape());
  }

  return DISPATCH_ALL_FIELDS(field, "cheetah.ring_cast", [&]() {
    using from_ring2k_t = ring2k_t;
    return DISPATCH_ALL_FIELDS(ftype, "cheetah.ring_cast", [&]() {
      using to_ring2k_t = ring2k_t;
      NdArrayView<const from_ring2k_t> _in(in);
      NdArrayView<to_ring2k_t> _res(res);
      pforeach(0, numel, [&](int64_t idx) {
        _res[idx] = static_cast<to_ring2k_t>(_in[idx]);
      });
      return res;
    });
  });
}

// x < 0, |x| < threshold and x >= threshold
[[maybe_unused]] std::array<spu::Value, 3> ThreeCompareInSmallerRing(
    spu::SPUContext* ctx, const spu::Value& _x, float threshold,
    spu::FieldType working_ft) {
  namespace sk = spu::kernel;
  auto src_field = ctx->config().field();

  spu::Value x(CastRing(_x.data(), working_ft), _x.dtype());
  // FIXME(lwj): dirty hack
  const_cast<spu::RuntimeConfig*>(&ctx->config())->set_field(working_ft);
  ctx->getState<spu::mpc::Z2kState>()->setField(working_ft);

  const auto ONE = sk::hal::_constant(ctx, 1, x.shape());
  const auto True = sk::hal::_and(ctx, ONE, ONE);
  auto epsilon = sk::hal::epsilon(ctx, x.dtype(), x.shape());

  auto is_neg = sk::hlo::Less(
      ctx, x, sk::hal::constant(ctx, {0.0F}, x.dtype(), x.shape()));
  auto abs_x = sk::hal::_mux(ctx, is_neg, sk::hal::_negate(ctx, x), x)
                   .setDtype(x.dtype());

  auto is_inside_range = sk::hlo::Less(
      ctx, abs_x,
      sk::hal::constant(ctx, {threshold}, x.dtype(), x.shape()));  // |x| < t

  auto is_too_large = sk::hal::_xor(
      ctx, True, sk::hal::_or(ctx, is_neg, is_inside_range));  // x > t
  // FIXME(lwj): dirty hack
  const_cast<spu::RuntimeConfig*>(&ctx->config())->set_field(src_field);
  ctx->getState<spu::mpc::Z2kState>()->setField(src_field);

  is_neg = spu::Value(CastRing(is_neg.data(), src_field), spu::DT_I1);
  is_inside_range =
      spu::Value(CastRing(is_inside_range.data(), src_field), spu::DT_I1);
  is_too_large =
      spu::Value(CastRing(is_too_large.data(), src_field), spu::DT_I1);

  return {is_neg, is_inside_range, is_too_large};
}

}  // namespace

// logistic(x) = { 1e-4         if x < -7.0
//               { 1 - P^3(|x|) if x \in [-7.0, 0.0)
//               { P^3(|x|)     if x \in [0.0, 7.0)
//               { 1 - 1e-4     if x > 7.0
spu::Value Logistic(spu::SPUContext* ctx, const spu::Value& x) {
  namespace sk = spu::kernel;
  SPU_ENFORCE(x.isFxp());
  auto epsilon = sk::hal::epsilon(ctx, x.dtype(), x.shape());
  // x < 0, |x| <= 7.0, x > 7.0
  auto [is_neg, is_inside_range, is_too_large] =
      ThreeCompareInSmallerRing(ctx, x, 7.0F, spu::FM32);
  auto abs_x = sk::hal::_mux(ctx, is_neg, sk::hal::_negate(ctx, x), x)
                   .setDtype(x.dtype());

  const std::array<float, 4> P3 = {0.5040659140474179, 0.2705454505928517,
                                   -0.048738638679930966,
                                   0.0028663913027180167};
  const auto ONE = sk::hal::constant(ctx, {1.0F}, x.dtype(), x.shape());

  auto P3_eval = polynomial_3(ctx, abs_x, P3);

  // 1{|x| <= 7} ? P^3(|x|) : 1 - 1e-4
  auto ret =
      sk::hal::select(ctx, is_inside_range, P3_eval,
                      sk::hal::constant(ctx, {0.9999F}, x.dtype(), x.shape()));

  // 1{x < 0} ? 1 - sigmoid(|x|) : sigmoid(|x|)
  ret = sk::hal::select(ctx, is_neg, sk::hal::f_sub(ctx, ONE, ret), ret);
  return ret.setDtype(x.dtype());
}

// sigomoid(x) = 0.5 + 0.5*x * rsqrt(1 + x^2)
spu::Value Sigmoid(spu::SPUContext* ctx, const spu::Value& x) {
  namespace sk = spu::kernel;
  auto c05 = sk::hlo::Constant(ctx, 0.5F, x.shape());
  auto half = sk::hal::right_shift_arithmetic(ctx, x, 1);
  auto divisor = sk::hlo::Add(ctx, sk::hlo::Constant(ctx, 1, x.shape()),
                              sk::hal::f_square(ctx, x));
  return sk::hlo::Add(ctx, c05,
                      sk::hlo::Mul(ctx, half, Rsqrt(ctx, divisor, 1)));
}

spu::Value polynomial_3(spu::SPUContext* ctx, const spu::Value& x,
                        absl::Span<spu::Value const> coeffs) {
  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE_EQ(coeffs.size(), 4UL);
  auto x2 = spu::kernel::hal::f_square(ctx, x);
  auto x3 = spu::kernel::hal::f_mul(ctx, x, x2);

  // NOTE(lwj): lazy truncation
  auto P3_3 = spu::kernel::hal::_mul(ctx, x3, coeffs[3]);
  auto P3_2 = spu::kernel::hal::_mul(ctx, x2, coeffs[2]);
  auto P3_1 = spu::kernel::hal::_mul(ctx, x, coeffs[1]);

  return spu::kernel::hal::_add(
             ctx, coeffs[0],
             spu::kernel::hal::_trunc(
                 ctx, spu::kernel::hal::_add(
                          ctx, P3_3, spu::kernel::hal::_add(ctx, P3_2, P3_1))))
      .setDtype(x.dtype());
}
}  // namespace squirrel
