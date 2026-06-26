#include "libspu/mpc/flp/fp_add.h"

#include "libspu/core/type.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/flp/fp_check.h"
#include "libspu/mpc/flp/round_and_check_fp_api.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::flp {
namespace {

// MUX with 1-bit BShare control: result = cond ? true_val : false_val
// Uses mul_a1b for efficiency (no b2a conversion).
Value MuxA(SPUContext* ctx, const Value& cond_b, const Value& x_true,
           const Value& x_false) {
  auto diff = add_ss(ctx, x_true, negate_s(ctx, x_false));
  auto masked = mul_a1b(ctx, diff, cond_b);
  return add_ss(ctx, x_false, masked);
}

// Barrel left shift: result = x << amt  (amt is secret AShare, unsigned)
Value BarrelShiftLeft(SPUContext* ctx, const Value& x, const Value& amt,
                      int amt_bits) {
  Shape shape = x.shape();

  // Decompose amt into BShare bits (a2b_bits: BShare ring_reduce_ is safe)
  auto amt_bits_val = dynDispatch<Value>(ctx, "a2b_bits", amt,
                                         static_cast<int64_t>(amt_bits));

  Value result = x.clone();

  for (int i = 0; i < amt_bits; i++) {
    // Extract bit i of amt as effective 1-bit BShare
    auto shifted_i = rshift_b(ctx, amt_bits_val, {i});
    auto bit_i = and_bp(ctx, shifted_i, make_p(ctx, uint128_t{1}, shape));

    // Conditionally shift result by 2^i
    auto shifted = lshift_s(ctx, result, {static_cast<int64_t>(1 << i)});

    result = MuxA(ctx, bit_i, shifted, result);
  }
  return result;
}

// MSNZB + normalization in a single pass.
// Returns {m_norm, e_adj}.  m_norm is mantissa after left-shifting to
// bring MSB to position 2q+1, modulo 2^{2q+2}.  e_adj = e + k - q.
std::pair<Value, Value> NormalizeMantissa(SPUContext* ctx, const Value& m_sum,
                                           const Value& e, int bit_width,
                                           int q) {
  Shape shape = m_sum.shape();

  // Decompose m_sum into BShare bits (a2b_bits: BShare ring_reduce_ is safe)
  auto x_bits = dynDispatch<Value>(ctx, "a2b_bits", m_sum,
                                   static_cast<int64_t>(bit_width));

  auto zero_b = p2b(ctx, make_p(ctx, uint128_t{0}, shape));
  auto found = zero_b;

  auto pub_one = make_p(ctx, uint128_t{1}, shape);

  Value m_norm = p2s(ctx, make_p(ctx, uint128_t{0}, shape));
  Value e_adj = p2s(ctx, make_p(ctx, uint128_t{0}, shape));

  for (int i = bit_width - 1; i >= 0; i--) {
    // Extract bit i (BShare local op, no OT)
    auto bit_i = and_bp(ctx, rshift_b(ctx, x_bits, {i}), pub_one);

    // is_msb_i = bit_i AND NOT(found)
    auto not_found = xor_bp(ctx, found, pub_one);
    auto is_msb_i = and_bb(ctx, bit_i, not_found);
    found = xor_bb(ctx, found, is_msb_i);

    // Pre-compute normalized mantissa for MSNZB position i
    int shift_amt = 2 * q + 1 - i;
    Value m_opt;
    if (shift_amt > 0) {
      m_opt = lshift_s(ctx, m_sum, {static_cast<int64_t>(shift_amt)});
    } else if (shift_amt < 0) {
      m_opt = trunc_s(ctx, m_sum, static_cast<size_t>(-shift_amt),
                      SignType::Positive);
    } else {
      m_opt = m_sum.clone();
    }

    auto delta_p = make_p(
        ctx, static_cast<uint128_t>(static_cast<uint64_t>(i - q)), shape);
    auto e_opt = add_sp(ctx, e, delta_p);

    m_norm = MuxA(ctx, is_msb_i, m_opt, m_norm);
    e_adj = MuxA(ctx, is_msb_i, e_opt, e_adj);
  }

  return {m_norm, e_adj};
}

}  // namespace

SharedFloat FPAdd(SPUContext* ctx, const SharedFloat& lhs,
                  const SharedFloat& rhs) {
  SPU_ENFORCE(ctx != nullptr);
  SPU_ENFORCE(lhs.p == rhs.p && lhs.q == rhs.q,
              "format mismatch lhs.p={} lhs.q={} rhs.p={} rhs.q={}", lhs.p,
              lhs.q, rhs.p, rhs.q);
  SPU_ENFORCE(lhs.p > 0 && lhs.q > 0, "invalid format p={} q={}", lhs.p, lhs.q);
  SPU_ENFORCE(lhs.z.shape() == rhs.z.shape());
  SPU_ENFORCE(lhs.z.shape() == lhs.e.shape());

  const int p = lhs.p;
  const int q = lhs.q;
  const Shape& shape = lhs.z.shape();

  // Step 1: compare exponents
  // e_LT = 1 if lhs.e < rhs.e
  auto diff_e1 = add_ss(ctx, lhs.e, negate_s(ctx, rhs.e));
  auto e_LT = msb_s(ctx, diff_e1);
  // e_EQ = 1 if lhs.e == rhs.e
  auto e_EQ_opt = equal_ss(ctx, lhs.e, rhs.e);
  SPU_ENFORCE(e_EQ_opt.has_value(), "equal_ss not available");
  auto e_EQ = e_EQ_opt.value();

  // Step 2: m_LT = 1 if lhs.m < rhs.m
  auto diff_m = add_ss(ctx, lhs.m, negate_s(ctx, rhs.m));
  auto m_LT = msb_s(ctx, diff_m);

  // Step 3: swap condition
  // swap = e_LT OR (e_EQ AND m_LT) -> put larger in large, smaller in small
  auto swap_b = xor_bb(ctx, e_LT, and_bb(ctx, e_EQ, m_LT));  // e_LT XOR (e_EQ AND m_LT)

  auto large_z = MuxA(ctx, swap_b, rhs.z, lhs.z);
  auto large_s = MuxA(ctx, swap_b, rhs.s, lhs.s);
  auto large_e = MuxA(ctx, swap_b, rhs.e, lhs.e);
  auto large_m = MuxA(ctx, swap_b, rhs.m, lhs.m);

  auto small_z = MuxA(ctx, swap_b, lhs.z, rhs.z);
  auto small_s = MuxA(ctx, swap_b, lhs.s, rhs.s);
  auto small_e = MuxA(ctx, swap_b, lhs.e, rhs.e);
  auto small_m = MuxA(ctx, swap_b, lhs.m, rhs.m);

  // Step 4: d = large.e - small.e
  auto d = add_ss(ctx, large_e, negate_s(ctx, small_e));

  // Step 5-6: if d > q+1, skip alignment -> return large
  auto C_qp1 = make_p(ctx, static_cast<uint128_t>(q + 1), shape);
  auto diff_d = add_sp(ctx, negate_s(ctx, d), C_qp1);  // (q+1) - d
  auto cond_skip = msb_s(ctx, diff_d);                   // 1 if d > q+1

  // Main path: Steps 8-16
  const int d_bits = q + 2;  // d <= q+1, so d needs ceil(log2(q+1)) + 1 bits

  // Step 8: m_large_shifted = large.m << d
  auto m_large_shifted = BarrelShiftLeft(ctx, large_m, d, d_bits);

  // Step 9: small.m already in FM64, upper bits naturally zero
  auto m_small_extended = small_m;

  // Step 10: conditional negation if signs differ
  // sx = large.s XOR small.s
  auto sign_eq_opt = equal_ss(ctx, large_s, small_s);
  SPU_ENFORCE(sign_eq_opt.has_value());
  auto sign_eq = sign_eq_opt.value();
  auto sign_diff = xor_bp(ctx, sign_eq, make_p(ctx, uint128_t{1}, shape));
  auto m_small_signed = MuxA(ctx, sign_diff,
                              negate_s(ctx, m_small_extended), m_small_extended);

  // Step 11: m_sum = m_large_shifted + m_small_signed, width 2q+3
  auto m_sum = add_ss(ctx, m_large_shifted, m_small_signed);

  // Step 12: e_result = small.e (smaller exponent)
  auto e_result = small_e;

  // Step 13-15: MSNZB + normalize
  const int sum_bitwidth = static_cast<int>(2 * q + 3);
  auto [m_norm, e_norm] =
      NormalizeMantissa(ctx, m_sum, e_result, sum_bitwidth, q);

  // Step 16: round mantissa to q+1 bits
  SharedFloat temp_round(/*z=*/make_p(ctx, uint128_t{0}, shape),
                          /*s=*/make_p(ctx, uint128_t{0}, shape), e_norm, m_norm,
                          p, q);
  auto rounded = RoundMantissaAndCheckSharedApi(ctx, temp_round, 2 * q + 1, q);

  // Step 17: z = 1{m == 0}
  auto m_is_zero_opt =
      equal_sp(ctx, rounded.m, make_p(ctx, uint128_t{0}, shape));
  SPU_ENFORCE(m_is_zero_opt.has_value());
  auto m_is_zero = m_is_zero_opt.value();
  auto z_a = b2a(ctx, m_is_zero);

  // Step 18: s = large.s
  auto s_result = large_s;

  // Step 19: FPCheck with computed z, s, rounded e/m
  SharedFloat sf_add(z_a, s_result, rounded.e, rounded.m, p, q);
  auto out = FPCheck(ctx, sf_add);

  // MUX between skip path (return large) and normal path (out)
  auto z_final = MuxA(ctx, cond_skip, large_z, out.z);
  auto s_final = MuxA(ctx, cond_skip, large_s, out.s);
  auto e_final = MuxA(ctx, cond_skip, large_e, out.e);
  auto m_final = MuxA(ctx, cond_skip, large_m, out.m);

  return SharedFloat(z_final, s_final, e_final, m_final, p, q);
}

}  // namespace spu::mpc::flp

