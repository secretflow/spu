#include "libspu/mpc/flp/round_and_check_fp_api.h"

#include <cstdint>

#include "libspu/mpc/ab_api.h"

namespace spu::mpc::flp {

// Implements ΠRound&Check from BEACON (Figure 9).
//   Input:  mantissa m with (Q+1) bits, exponent e with (p+2) bits
//   Output: mantissa m' with (q+1) bits, exponent e' with (p+2) bits
//
//   Step 1: ⟨c⟩ = ΠLT(⟨m⟩_{Q+1}, 2^{Q+1} − 2^{Q−q−1})
//   Step 2: ⟨mc⟩_{q+1} = ΠRN^{Q+1,Q−q}(⟨m⟩_{Q+1})
//   Step 3: ⟨m'⟩_{q+1} = ΠMUX(⟨c⟩_B, ⟨mc⟩_{q+1}, 2^q)
//   Step 4: ⟨e'⟩_{p+2} = ΠMUX(⟨c⟩_B, ⟨e⟩_{p+2}, ⟨e⟩_{p+2} + 1)
//
//   ΠMUX semantics: ΠMUX(c, x, y) = c ? x : y  (c=1 selects x)
//   ΠRN(x, s) = (x + 2^{s-1}) >> s  (round-to-nearest, ties-up)
//
SharedFloat RoundMantissaAndCheckSharedApi(SPUContext* ctx,
                                           const SharedFloat& in, int Q,
                                           int q) {
  const int s = Q - q;
  const FieldType field = in.m.storage_type().as<Ring2k>()->field();

  const uint128_t round_bias = uint128_t{1} << (s - 1);
  const uint128_t threshold = (uint128_t{1} << (Q + 1)) -
                               (uint128_t{1} << (Q - q - 1));

  auto bias_p = make_p(ctx, round_bias, in.m.shape(), field);
  auto threshold_p = make_p(ctx, threshold, in.m.shape(), field);

  // Step 2: ΠRN — round-to-nearest (ties-up)
  //   m_c = (m + 2^{s-1}) >> s
  auto m_plus = add_ss(ctx, in.m, p2s(ctx, bias_p));
  auto m_c = trunc2_s(ctx, m_plus, static_cast<size_t>(s),
                      SignType::Unknown, true, true);

  // Renormalized mantissa for overflow case: m_c >> 1 = 2^q (fits in q+1 bits)
  auto m_renorm = trunc2_s(ctx, m_c, 1, SignType::Unknown, true, true);

  // Step 1: ΠLT — compare m against threshold
  //   cmp = m - threshold in ring arithmetic
  //   MSB=1 when m < threshold (unsigned wrap-around = negative in 2's complement)
  //   MSB=0 when m >= threshold
  auto cmp_x = add_ss(ctx, in.m, negate_s(ctx, p2s(ctx, threshold_p)));
  auto msb_bit_full = msb_s(ctx, cmp_x);
  // Extract LSB (all bits are identical since msb_s returns all-bits=MSB)
  auto c_bshare = and_bp(ctx, msb_bit_full,
                         make_p(ctx, 1, in.m.shape(), field));

  // need_overflow = NOT(c) = 1 when m >= threshold (overflow case)
  // Paper: c=1 → no overflow (m < threshold), c=0 → overflow (m >= threshold)
  //        need_overflow = 1-c = NOT(c)
  auto need_overflow = xor_bp(ctx, c_bshare,
                              make_p(ctx, 1, in.m.shape(), field));

  // Step 3: ΠMUX — select mantissa
  //   c=1 (no overflow):  m' = m_c  (RN result)
  //   c=0 (overflow):     m' = 2^q  (renormalized)
  //
  //   In terms of need_overflow:
  //     need_overflow=0 (no overflow): m' = m_c
  //     need_overflow=1 (overflow):    m' = m_renorm (= m_c >> 1 = 2^q)
  //
  //   Using arithmetic: m' = m_c - (m_c - m_renorm) * need_overflow
  auto diff = add_ss(ctx, m_c, negate_s(ctx, m_renorm));
  auto m_prime = add_ss(ctx, m_c,
                        negate_s(ctx, mul_a1b(ctx, diff, need_overflow)));

  // Step 4: ΠMUX — select exponent
  //   c=1 (no overflow): e' = e
  //   c=0 (overflow):    e' = e + 1
  //
  //   In terms of need_overflow:
  //     need_overflow=0 (no overflow): e' = e
  //     need_overflow=1 (overflow):    e' = e + 1
  auto e_prime = add_ss(ctx, in.e,
                        mul_a1b(ctx,
                                p2s(ctx, make_p(ctx, 1, in.e.shape(), field)),
                                need_overflow));

  SharedFloat out;
  out.z = in.z;
  out.s = in.s;
  out.e = e_prime;
  out.m = m_prime;
  out.p = in.p;
  out.q = q;

  return out;
}

}  // namespace spu::mpc::flp
