#include "libspu/mpc/flp/fp_mul.h"

#include "libspu/core/type.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/flp/fp_check.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::flp {
namespace {

// A-share bit OR for 0/1 arithmetic shares:
//   x OR y = x + y - x*y
Value BitOrA(SPUContext* ctx, const Value& x, const Value& y) {
  auto xy = mul_ss(ctx, x, y);
  auto sum = add_ss(ctx, x, y);
  return add_ss(ctx, sum, negate_s(ctx, xy));
}

// A-share bit XOR for 0/1 arithmetic shares:
//   x XOR y = x + y - 2*x*y
Value BitXorA(SPUContext* ctx, const Value& x, const Value& y) {
  auto xy = mul_ss(ctx, x, y);
  auto two_xy = add_ss(ctx, xy, xy);
  auto sum = add_ss(ctx, x, y);
  return add_ss(ctx, sum, negate_s(ctx, two_xy));
}

// MUX for arithmetic shares controlled by a 1-bit B-share:
//   cond = 1 -> x_true
//   cond = 0 -> x_false
//
//   out = x_false + cond * (x_true - x_false)
Value MuxA(SPUContext* ctx, const Value& cond_b, const Value& x_true,
           const Value& x_false) {
  auto diff = add_ss(ctx, x_true, negate_s(ctx, x_false));
  auto masked = mul_a1b(ctx, diff, cond_b);
  return add_ss(ctx, x_false, masked);
}

Value MakePublicWithFxp(SPUContext* ctx, uint128_t val, const Shape& shape,
                        FieldType field, int64_t fxp_bits) {
  auto pub = make_p(ctx, val, shape, field);
  if (fxp_bits > 0) {
    pub.data().set_fxp_bits(fxp_bits);
  }
  return pub;
}

// RNTE-style rounding used by the paper:
//   RNTE(x, shift) ~= (x + 2^{shift-1}) >> shift
//
// x is a non-negative mantissa product, so use unsigned exact truncation.
Value RNTE(SPUContext* ctx, const Value& x, size_t shift_bits,
           size_t out_bits) {
  SPU_ENFORCE(shift_bits > 0);

  const auto field = x.storage_type().as<Ring2k>()->field();
  const auto in_bits = x.data().fxp_bits() > 0
                           ? x.data().fxp_bits()
                           : static_cast<int64_t>(8 * SizeOf(field));

  auto bias_p = MakePublicWithFxp(ctx, uint128_t{1} << (shift_bits - 1),
                                  x.shape(), field, in_bits);

  auto x_plus = add_ss(ctx, x, p2s(ctx, bias_p));
  x_plus.data().set_fxp_bits(in_bits);

  auto y = trunc2_s(ctx, x_plus, shift_bits, SignType::Positive,
                    /*exact=*/true,
                    /*signed_arith=*/false);

  // Canonicalize the RNTE output into the target mantissa ring.
  ring_reduce_(y.data(), out_bits);
  y.data().set_fxp_bits(static_cast<int64_t>(out_bits));

  return y;
}

// Compute c = 1{m < threshold} as a 1-bit B-share.
//
// Paper line:
//   c = GT(2^{2q+1} - 2^{q-1}, m)
//
// Equivalent:
//   c = MSB(m - threshold)
Value LessThanPublicByMsb(SPUContext* ctx, const Value& x, uint128_t threshold,
                          size_t bitwidth) {
  const auto field = x.storage_type().as<Ring2k>()->field();

  auto threshold_p = MakePublicWithFxp(ctx, threshold, x.shape(), field,
                                       static_cast<int64_t>(bitwidth));
  auto threshold_s = p2s(ctx, threshold_p);

  auto diff = add_ss(ctx, x, negate_s(ctx, threshold_s));
  diff.data().set_fxp_bits(static_cast<int64_t>(bitwidth));

  auto msb_full = msb_s(ctx, diff);

  // mul_a1b requires 1-bit B-share.
  auto one_p = make_p(ctx, uint128_t{1}, x.shape(), field);
  return and_bp(ctx, msb_full, one_p);
}

}  // namespace

SharedFloat FPMul(SPUContext* ctx, const SharedFloat& lhs,
                  const SharedFloat& rhs) {
  SPU_ENFORCE(ctx != nullptr);

  SPU_ENFORCE(lhs.p == rhs.p, "lhs.p={} rhs.p={} mismatch", lhs.p, rhs.p);
  SPU_ENFORCE(lhs.q == rhs.q, "lhs.q={} rhs.q={} mismatch", lhs.q, rhs.q);
  SPU_ENFORCE(lhs.p > 0 && lhs.q > 0, "invalid float format p={} q={}", lhs.p,
              lhs.q);

  SPU_ENFORCE(lhs.z.shape() == rhs.z.shape());
  SPU_ENFORCE(lhs.s.shape() == rhs.s.shape());
  SPU_ENFORCE(lhs.e.shape() == rhs.e.shape());
  SPU_ENFORCE(lhs.m.shape() == rhs.m.shape());

  SPU_ENFORCE(lhs.z.shape() == lhs.s.shape());
  SPU_ENFORCE(lhs.z.shape() == lhs.e.shape());
  SPU_ENFORCE(lhs.z.shape() == lhs.m.shape());

  const int p = lhs.p;
  const int q = lhs.q;

  const size_t mantissa_bw = static_cast<size_t>(q + 1);
  const size_t prod_bw = static_cast<size_t>(2 * q + 2);

  SPU_ENFORCE(prod_bw <= 64, "mantissa product width={} exceeds FM64", prod_bw);

  // 1. e = lhs.e + rhs.e
  auto e = add_ss(ctx, lhs.e, rhs.e);

  // 2. m = lhs.m * rhs.m, output width 2q + 2.
  auto lhs_m = lhs.m.clone();
  auto rhs_m = rhs.m.clone();

  // Mantissa lives in q+1-bit ring.
  ring_reduce_(lhs_m.data(), mantissa_bw);
  ring_reduce_(rhs_m.data(), mantissa_bw);

  lhs_m.data().set_fxp_bits(static_cast<int64_t>(mantissa_bw));
  rhs_m.data().set_fxp_bits(static_cast<int64_t>(mantissa_bw));

  auto m_prod = mix_mul_ss(ctx, lhs_m, rhs_m, SignType::Unknown,
                           SignType::Unknown, FM64, prod_bw,
                           /*signed_arith=*/false);

  // Product lives in 2q+2-bit ring.
  ring_reduce_(m_prod.data(), prod_bw);
  m_prod.data().set_fxp_bits(static_cast<int64_t>(prod_bw));

  // 3. c = GT(2^{2q+1} - 2^{q-1}, m)
  //
  // c = 1 means product is in the lower normalized range:
  // choose RNTE(m, q), exponent unchanged.
  const uint128_t threshold =
      (uint128_t{1} << (2 * q + 1)) - (uint128_t{1} << (q - 1));

  auto c = LessThanPublicByMsb(ctx, m_prod, threshold, prod_bw);

  // 4. m1 = RNTE(m, q) mod 2^{q+1}
  auto m1 = RNTE(ctx, m_prod, static_cast<size_t>(q), mantissa_bw);

  // 5. m2 = RNTE(m, q + 1)
  auto m2 = RNTE(ctx, m_prod, static_cast<size_t>(q + 1), mantissa_bw);

  // 6. e2 = e + 1
  const auto e_field = e.storage_type().as<Ring2k>()->field();
  auto one_e = make_p(ctx, uint128_t{1}, e.shape(), e_field);
  auto e2 = add_sp(ctx, e, one_e);

  // 7. m = MUX(c, m1, m2)
  auto m = MuxA(ctx, c, m1, m2);

  // Final mantissa must be canonical q+1-bit value.
  ring_reduce_(m.data(), mantissa_bw);
  m.data().set_fxp_bits(static_cast<int64_t>(mantissa_bw));

  // 8. e = MUX(c, e, e2)
  auto e_final = MuxA(ctx, c, e, e2);

  // 9. s = lhs.s XOR rhs.s
  auto s = BitXorA(ctx, lhs.s, rhs.s);

  // 10. z = lhs.z OR rhs.z
  auto z = BitOrA(ctx, lhs.z, rhs.z);

  // 11. FPCheck(z, s, e, m)
  auto out = FPCheck(ctx, SharedFloat(z, s, e_final, m, p, q));

  // FPCheck may preserve arithmetic high bits in m, so canonicalize again.
  ring_reduce_(out.m.data(), mantissa_bw);
  out.m.data().set_fxp_bits(static_cast<int64_t>(mantissa_bw));

  return out;
}

}  // namespace spu::mpc::flp