#include "libspu/mpc/flp/fp_check.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"

namespace spu::mpc::flp {

namespace {

// Make a public constant from unsigned value
Value MakePub(SPUContext* ctx, uint64_t val, const Shape& shape) {
  return make_p(ctx, static_cast<uint128_t>(val), shape);
}

// Make a public constant from signed value (for ring repr of negatives)
Value MakePubSigned(SPUContext* ctx, int64_t val, const Shape& shape) {
  return make_p(ctx, static_cast<uint128_t>(static_cast<uint64_t>(val)), shape);
}

// MUX where target is public: result = old + cond × (new_pub - old)
// cond is BShare, old is AShare, new_pub is Public
Value MuxPub(SPUContext* ctx, const Value& cond, const Value& old,
             const Value& new_pub) {
  auto cond_a = b2a(ctx, cond);
  auto neg_old = negate_s(ctx, old);
  auto diff = add_sp(ctx, neg_old, new_pub);  // new - old
  auto delta = mul_ss(ctx, cond_a, diff);
  return add_ss(ctx, old, delta);
}

}  // namespace

SharedFloat FPCheck(SPUContext* ctx, const SharedFloat& sf) {
  const int p = sf.p;
  const int q = sf.q;
  SPU_ENFORCE(p > 0 && q > 0, "p={} q={} must be positive", p, q);

  const Shape& shape = sf.z.shape();
  SPU_ENFORCE(
      sf.s.shape() == shape && sf.e.shape() == shape && sf.m.shape() == shape,
      "all components must share the same shape");

  // Step 1: derive constants from p, q
  // In FM64, negative values are passed via uint64_t wrap-around
  const int64_t overflow_thresh = (1LL << (p - 1)) - 1;  // 2^{p-1} - 1
  const int64_t max_e = 1LL << (p - 1);                  // 2^{p-1}
  const int64_t min_e = 1 - (1LL << (p - 1));            // 1 - 2^{p-1}
  const uint64_t max_m = 1ULL << q;                      // 2^q

  auto C_one = MakePub(ctx, 1, shape);
  auto C_ov_thresh = MakePubSigned(ctx, overflow_thresh, shape);  // 127
  auto C_126 = MakePub(ctx, 126, shape);
  auto C_max_m = MakePub(ctx, max_m, shape);
  auto C_max_e = MakePubSigned(ctx, max_e, shape);
  auto C_min_e = MakePubSigned(ctx, min_e, shape);

  // Step 2: cond_ov = (e > overflow_thresh)
  // e > T ⇔ T - e < 0 ⇔ msb_s(T + (-e)) == 1
  auto diff_ov = add_sp(ctx, negate_s(ctx, sf.e), C_ov_thresh);
  auto cond_ov = msb_s(ctx, diff_ov);

  // Step 3: cond_z = (z == 1)
  auto cond_z_opt = equal_sp(ctx, sf.z, C_one);
  SPU_ENFORCE(cond_z_opt.has_value(), "equal_sp not available");
  auto cond_z = cond_z_opt.value();

  // Step 4: cond_e_under = (e < underflow_thresh) i.e. (e < 2 - 2^{p-1})
  // e < -126 ⇔ e + 126 < 0 ⇔ msb_s(e + 126) == 1
  auto biased = add_sp(ctx, sf.e, C_126);
  auto cond_e_under = msb_s(ctx, biased);

  // Step 5: cond_un = cond_z OR cond_e_under
  // a OR b = NOT(NOT(a) AND NOT(b))
  auto not_z = xor_bp(ctx, cond_z, C_one);
  auto not_ue = xor_bp(ctx, cond_e_under, C_one);
  auto and_not = and_bb(ctx, not_z, not_ue);
  auto cond_un = xor_bp(ctx, and_not, C_one);

  // Step 6: MUX with cond_ov → clamp m, e to max
  auto m_after_ov = MuxPub(ctx, cond_ov, sf.m, C_max_m);
  auto e_after_ov = MuxPub(ctx, cond_ov, sf.e, C_max_e);

  // Step 7: MUX with cond_un → clamp z=1, m=0, e=min_e
  auto C_zero = MakePub(ctx, 0, shape);
  auto z_prime = MuxPub(ctx, cond_un, sf.z, C_one);
  auto m_prime = MuxPub(ctx, cond_un, m_after_ov, C_zero);
  auto e_prime = MuxPub(ctx, cond_un, e_after_ov, C_min_e);

  return SharedFloat(z_prime, sf.s, e_prime, m_prime, p, q);
}

}  // namespace spu::mpc::flp
