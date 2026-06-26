#include "libspu/mpc/flp/fp_div.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/nonlinear/ext_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/flp/fp_check.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::flp {
namespace {

Value Pub(SPUContext* ctx, uint64_t v, const Shape& sh, FieldType f) {
  return make_p(ctx, static_cast<uint128_t>(v), sh, f);
}

Value MulTrunc(SPUContext* ctx, const Value& x, const Value& y,
               size_t shift) {
  auto prod = mul_ss(ctx, x, y);
  if (shift == 0) return prod;
  return trunc2_s(ctx, prod, shift, SignType::Unknown, true, true);
}

// Zero-extend a Value's logical bit width.
// When src_ring == dst_ring (same ring), the extension is automatic:
// since the ring is larger than src_bits, the high bits are naturally zero.
// RingExtendProtocol is designed for cross-ring extension (FM32→FM64)
// where it properly handles the wrap and ring-type transition.
// For same-ring extension, no OT protocol is needed.
Value ZXtValue(SPUContext* ctx, const Value& v,
               int src_bits, int dst_bits) {
  (void)ctx;
  (void)src_bits;
  (void)dst_bits;
  return v;
}

}  // anonymous namespace

SharedFloat FPDiv(SPUContext* ctx, const SharedFloat& num,
                  const SharedFloat& den,
                  FieldType exp_field,
                  FieldType man_field) {
  SPU_ENFORCE(num.p == den.p && num.q == den.q);
  const int p = num.p, q = num.q;
  const Shape& sh = num.m.shape();
  const FieldType fd = num.m.storage_type().as<Ring2k>()->field();
  const int man_bits = q + 1;

  auto one = Pub(ctx, 1, sh, fd);

  // ==============================================================
  // Steps 1-3: exponent diff & mantissa normalisation (FFPDiv)
  //   m1 = ZXt(α1.m, q+2); m2 = α2.m; e = α1.e − α2.e
  //   if α1.m < α2.m: m1 = 2·m1; e = e − 1
  // ==============================================================
  Value m1 = num.m;
  Value m2 = den.m;
  Value e  = add_ss(ctx, num.e, negate_s(ctx, den.e));

  auto lt_msb = msb_s(ctx, add_ss(ctx, m1, negate_s(ctx, m2)));
  auto lt     = and_bp(ctx, lt_msb, one);
  m1 = add_ss(ctx, m1, mul_a1b(ctx, m1, lt));
  e  = add_ss(ctx, e, negate_s(ctx, b2a(ctx, lt)));

  // ==============================================================
  // Step 4: Newton parameters (paper §V-E)
  //   t = 2;  g = ⌈(q+1)/(2t)⌉ + 1;  k0 = g + 1
  //   k_i = 2^i·(g−1) + 3   for i = 1 .. t
  // ==============================================================
  const int t = 2;
  const int g = static_cast<int>(
      std::ceil(static_cast<double>(q + 1) / static_cast<double>(2 * t))) + 1;
  // k_i = 2^i*(g-1) + 3   for i = 1 .. t;  k0 is defined separately
  auto ki = [g](int i) { return (1 << i) * (g - 1) + 3; };
  const int k0 = g + 1;

  // ==============================================================
  // Steps 5-6: Lookup table for reciprocal of m2
  //   h = TR(m2, q−g) mod 2^g   →   r0 (k0+2 = g+3 bits)
  //   r0 = Lrecp_init(h) ≈ 2^{g+2} / (1 + h/2^g)
  // ==============================================================
  Value h_idx_raw = (q > g)
      ? trunc2_s(ctx, m2, static_cast<size_t>(q - g),
                 SignType::Unknown, true, true)
      : m2;

  // FIX: Remove the leading-1 bit: h = (m2 >> (q-g)) - 2^g
  // This gives h ∈ [0, 2^g) for normalized inputs m2 ∈ [2^q, 2^{q+1})
  auto leading_one_p = Pub(ctx, uint64_t{1} << g, sh, fd);
  auto h_idx = add_ss(ctx, h_idx_raw, negate_s(ctx, p2s(ctx, leading_one_p)));

  const size_t tbl_sz = static_cast<size_t>(1) << g;
  SPU_ENFORCE(tbl_sz <= 256, "2^g=%zu > 256", tbl_sz);

  NdArrayRef tbl_buf(makeType<RingTy>(fd), {static_cast<int64_t>(tbl_sz)});
  auto tv = NdArrayView<uint64_t>(tbl_buf);
  const size_t r0_bits = static_cast<size_t>(k0 + 2);
  const uint64_t max_r0 = (r0_bits < 64)
      ? ((uint64_t{1} << r0_bits) - 1) : UINT64_MAX;
  const double scale = std::ldexp(1.0, g + 2);

  for (size_t i = 0; i < tbl_sz; ++i) {
    double d = 1.0 + static_cast<double>(i) / static_cast<double>(tbl_sz);
    tv[static_cast<int64_t>(i)] = std::min(
        static_cast<uint64_t>(std::llround(scale / d)), max_r0);
  }
  auto tbl_pub = tbl_buf.as(makeType<Pub2kTy>(fd));
  Value table_val(std::move(tbl_pub), DT_INVALID);

  Value r = lut_sp(ctx, h_idx, table_val, SizeOf(fd) * 8, fd);
  { auto rd = r.data(); rd.set_fxp_bits(0); r = Value(rd, r.dtype()); }

  // ==============================================================
  // ZXt m1, m2, r to wider logical width for Newton iterations.
  // The ring stays fd (e.g. FM64), but the logical bit width expands.
  // ==============================================================
  Value m2_w = ZXtValue(ctx, m2, man_bits, q + k0 + 2);
  Value m1_w = ZXtValue(ctx, m1, man_bits, q + k0 + 2);
  Value r_w  = ZXtValue(ctx, r,  static_cast<int>(r0_bits),
                        static_cast<int>(r0_bits + 1));

  // ==============================================================
  // Steps 7-10: Newton–Raphson iterations
  //   r_new = r · (2^{q+k_prev+2} − m₂·r) >> (q + 2·k_prev − k_new + 1)
  // ==============================================================
  for (int i = 1; i <= t; ++i) {
    const int kp = ki(i - 1);
    const int kn = ki(i);
    auto two_pow = Pub(ctx, uint64_t{1} << (q + kp + 2), sh, fd);

    auto prod = mul_ss(ctx, m2_w, r_w);
    auto f = add_ss(ctx, p2s(ctx, two_pow), negate_s(ctx, prod));
    size_t rshift = static_cast<size_t>(std::max(0, q + 2 * kp - kn + 1));
    r_w = MulTrunc(ctx, r_w, f, rshift);
  }

  // ==============================================================
  // Step 11:  Approximate quotient
  //   m00 = m1 ×_{k_t+q+2} r_t   (keep k_t+q+2 bits)
  //   m0  = TR(m00, k_t)          (→ q+2 bits)
  // ==============================================================
  const int kt = ki(t);
  const size_t prod_bits = static_cast<size_t>(q + kt + 2);
  const size_t keep_bits = static_cast<size_t>(kt + q + 2);
  auto m00 = (prod_bits > keep_bits)
      ? MulTrunc(ctx, m1_w, r_w, prod_bits - keep_bits)
      : mul_ss(ctx, m1_w, r_w);
  auto m0 = trunc2_s(ctx, m00, static_cast<size_t>(kt),
                     SignType::Unknown, true, true);

  // ==============================================================
  // Steps 12-14: ULP correction (paper §V-E)
  //   y1 = m2 ×_{q+3} m0     (keep q+3 bits)
  //   y2 = y1 + ZXt(m2, q+3)
  //   y  = (m1 ×_{q+3} 2^{q+1}) − (y1 + y2)
  //   (lt, eq) = LT&EQ(0, y)
  //   m = lt ⊕ (eq ∧ (m0 mod 2 = 1)) ? m0+1 : m0
  // ==============================================================
  // y1 = m2 ×_{q+3} m0
  auto y1 = MulTrunc(ctx, m2_w, m0, (q + 1 + q + 2) - (q + 3));

  // y2 = y1 + ZXt(m2, q+3): m2_w already has logical width q+k0+2 ≥ q+3
  auto y2 = add_ss(ctx, y1, m2_w);

  // target = m1 ×_{q+3} 2^{q+1}
  auto two_q1 = Pub(ctx, uint64_t{1} << (q + 1), sh, fd);
  auto target = MulTrunc(ctx, m1_w, p2s(ctx, two_q1),
                         (q + 2 + q + 2) - (q + 3));

  // y = target − (y1 + y2)
  auto y = add_ss(ctx, target, negate_s(ctx, add_ss(ctx, y1, y2)));

  auto lt_y = and_bp(ctx, msb_s(ctx, y), one);
  auto eq_opt = equal_sp(ctx, y, p2s(ctx, Pub(ctx, 0, sh, fd)));
  SPU_ENFORCE(eq_opt.has_value(), "equal_sp not available");
  auto eq_y = eq_opt.value();

  auto m0_half = trunc2_s(ctx, m0, 1, SignType::Unknown, true, true);
  auto m0_halved_dbl = add_ss(ctx, m0_half, m0_half);
  auto parity = add_ss(ctx, m0, negate_s(ctx, m0_halved_dbl));
  auto parity_zero_opt = equal_sp(ctx, parity, p2s(ctx, Pub(ctx, 0, sh, fd)));
  SPU_ENFORCE(parity_zero_opt.has_value(), "equal_sp for parity");
  auto is_even = parity_zero_opt.value();
  auto is_odd  = xor_bp(ctx, is_even, one);

  auto eq_and_odd = and_bb(ctx, eq_y, is_odd);
  auto not_lt     = xor_bp(ctx, lt_y, one);
  auto not_eao    = xor_bp(ctx, eq_and_odd, one);
  auto not_cond   = and_bb(ctx, not_lt, not_eao);
  auto cond       = xor_bp(ctx, not_cond, one);

  auto m_raw = add_ss(ctx, m0, b2a(ctx, cond));
  auto m_final = trunc2_s(ctx, m_raw, 1, SignType::Unknown, true, true);

  // ==============================================================
  // Step 15: sign, zero, and FPCheck
  //   s = α1.s ⊕ α2.s;  z = α1.z
  // ==============================================================
  auto s_final = xor_bb(ctx, a2b(ctx, num.s), a2b(ctx, den.s));
  auto z_final = num.z;

  SharedFloat raw;
  raw.z = z_final;
  raw.s = s_final;
  raw.e = e;
  raw.m = m_final;
  raw.p = p;
  raw.q = q;

  return FPCheck(ctx, raw);
}

}  // namespace spu::mpc::flp
