// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"

#include <utility>

#include "seal/util/iterator.h"
#include "seal/util/numth.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/uintarith.h"
#include "yacl/base/int128.h"

#include "libspu/mpc/cheetah/rlwe/utils.h"  // BarrettReduce
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

inline static uint64_t U64BitMask(size_t bw) {
  SPU_ENFORCE(bw > 0 && bw <= 64);
  return bw == 64 ? static_cast<uint64_t>(-1)
                  : (static_cast<uint64_t>(1) << bw) - 1;
}

inline static uint128_t AssignU128(uint64_t lo, uint64_t hi) {
  return (static_cast<uint128_t>(hi) << 64) | lo;
}

// return x^-1 mod 2^k for odd x
template <typename T>
constexpr T Inv2k(const T &x) {
  SPU_ENFORCE(x & 1, "need odd input");
  constexpr int nbits = sizeof(T) * 8;
  T inv = 1;
  T p = x;
  for (int i = 1; i < nbits; ++i) {
    inv *= p;
    p *= p;
  }
  return inv;
}

struct ModulusSwitchHelper::Impl {
 public:
  explicit Impl(uint32_t base_mod_bitlen, seal::SEALContext context)
      : base_mod_bitlen_(base_mod_bitlen), context_(std::move(context)) {
    SPU_ENFORCE(context_.parameters_set(), "invalid seal context");
    SPU_ENFORCE(base_mod_bitlen <= 128 && base_mod_bitlen >= 2,
                fmt::format("invalid base bitlen {}", base_mod_bitlen));
    Init();
  }

  inline seal::parms_id_type parms_id() const {
    return context_.key_parms_id();
  }

  inline uint32_t base_mod_bitlen() const { return base_mod_bitlen_; }

  inline uint32_t coeff_modulus_size() const { return Q_div_t_mod_qi_.size(); }

  // limbs mod 2^k
  static uint128_t ModLimbsRing2k(const uint64_t *limbs, size_t size,
                                  size_t mod_bit_width) {
    SPU_ENFORCE(mod_bit_width <= 128 && mod_bit_width >= 2);

    uint64_t num_limbs = (mod_bit_width + 63) / 64;
    size_t msb = mod_bit_width - 64 * (num_limbs - 1);
    uint64_t Q_mod_t_lo = limbs[0];
    uint64_t Q_mod_t_hi = size > 1 ? limbs[1] : 0;
    if (num_limbs > 1) {
      Q_mod_t_hi &= U64BitMask(msb);
    } else {
      Q_mod_t_hi = 0;
      Q_mod_t_lo &= U64BitMask(msb);
    }
    return AssignU128(Q_mod_t_lo, Q_mod_t_hi);
  }

  template <typename Scalar>
  void ModulusUpAt(ArrayView<Scalar> src, size_t mod_idx,
                   absl::Span<uint64_t> out) const {
    using namespace seal::util;
    SPU_ENFORCE(sizeof(Scalar) < sizeof(uint128_t));
    SPU_ENFORCE_EQ(sizeof(Scalar) * 8, absl::bit_ceil(base_mod_bitlen_),
                   fmt::format("expect base_mod_bitlen={} but got {}",
                               base_mod_bitlen_, sizeof(Scalar) * 8));
    const size_t n = src.numel();
    SPU_ENFORCE_EQ(n, out.size());
    size_t num_modulus = coeff_modulus_size();
    SPU_ENFORCE(mod_idx < num_modulus,
                fmt::format("ModulusUpAt: invalid mod_idx ({} >= {})", mod_idx,
                            num_modulus));
    const auto &modulus = context_.key_context_data()->parms().coeff_modulus();
    // round(Q/t*x) = k*x + round(r*x/t) where k = floor(Q/t), r = Q mod t
    // round(Q/t*x) mod qi = ((k mod qi)*x + round(r*x/t)) mod qi
    for (size_t i = 0; i < n; ++i) {
      // u = (Q mod t)*x mod qi
      Scalar x = src[i];
      uint64_t x64 = BarrettReduce(x, modulus[mod_idx]);
      uint64_t u = seal::util::multiply_uint_mod(x64, Q_div_t_mod_qi_[mod_idx],
                                                 modulus[mod_idx]);
      // uint128_t can conver uint32_t/uint64_t mult here
      Scalar v = ((Q_mod_t_ * x + t_half_) >> base_mod_bitlen_);
      out[i] = BarrettReduce(u + v, modulus[mod_idx]);
    }
  }

  // NOTE(juhou): we need 256-bit to store the product `x * (Q mod t)` for x, t
  // \in [2^64, 2^128).
  void ModulusUpAt(ArrayView<uint128_t> src, size_t mod_idx,
                   absl::Span<uint64_t> out) const {
    using namespace seal::util;
    SPU_ENFORCE_EQ(sizeof(uint128_t) * 8, absl::bit_ceil(base_mod_bitlen_),
                   fmt::format("expect base_mod_bitlen={} but got {}",
                               base_mod_bitlen_, sizeof(uint128_t) * 8));
    size_t n = src.numel();
    SPU_ENFORCE_EQ(n, out.size());
    size_t num_modulus = coeff_modulus_size();
    SPU_ENFORCE(mod_idx < num_modulus,
                fmt::format("ModulusUpAt: invalid mod_idx ({} >= {})", mod_idx,
                            num_modulus));
    const auto &modulus = context_.key_context_data()->parms().coeff_modulus();
    for (size_t i = 0; i < n; ++i) {
      uint128_t x = src[i];
      uint64_t x64 = BarrettReduce(x, modulus[mod_idx]);
      uint64_t u = seal::util::multiply_uint_mod(x64, Q_div_t_mod_qi_[mod_idx],
                                                 modulus[mod_idx]);
      using namespace seal::util;
      const auto *Q_mod_t = reinterpret_cast<const uint64_t *>(&Q_mod_t_);
      const auto *xlimbs = reinterpret_cast<const uint64_t *>(&x);
      const auto *t_half = reinterpret_cast<const uint64_t *>(&t_half_);
      constexpr size_t kU128Limbs = 2;

      // Compute round(x * Q_mod_t / t) for 2^64 < x, t <= 2^128
      // round(x * Q_mod_t / t) = floor((x * Q_mod_t + t_half) / t)
      // We need 4 limbs to store the product x * Q_mod_t
      std::vector<uint64_t> mul_limbs(2 * kU128Limbs);
      std::vector<uint64_t> add_limbs(2 * kU128Limbs);
      std::vector<uint64_t> rs_limbs(kU128Limbs + 1);
      multiply_uint(Q_mod_t, kU128Limbs, xlimbs, kU128Limbs, 2 * kU128Limbs,
                    mul_limbs.data());
      add_uint(mul_limbs.data(), 2 * kU128Limbs, t_half, kU128Limbs,
               /*carry*/ 0, 2 * kU128Limbs, add_limbs.data());
      // NOTE(juhou) base_mod_bitlen_ > 64, we can direct drop the LSB here.
      right_shift_uint192(add_limbs.data() + 1, base_mod_bitlen_ - 64,
                          rs_limbs.data());
      out[i] = BarrettReduce(u + AssignU128(rs_limbs[0], rs_limbs[1]),
                             modulus[mod_idx]);
    }
  }

  template <typename Scalar>
  void CenteralizeAt(ArrayView<Scalar> src, size_t mod_idx,
                     absl::Span<uint64_t> out) const {
    using namespace seal::util;
    SPU_ENFORCE_EQ(sizeof(Scalar) * 8, absl::bit_ceil(base_mod_bitlen_),
                   fmt::format("expect base_mod_bitlen={} but got {}",
                               base_mod_bitlen_, sizeof(Scalar) * 8));

    const auto &modulus = context_.key_context_data()->parms().coeff_modulus();
    SPU_ENFORCE(mod_idx < coeff_modulus_size(), "Centeralize: invalid mod_idx");
    size_t n = src.numel();
    SPU_ENFORCE(n == out.size(), "Centeralize: size mismatch");

    const auto &mod_qj = modulus[mod_idx];
    // view x \in [0, 2^k) as [-2^{k-1}, 2^{k-1})
    for (size_t i = 0; i < n; ++i) {
      auto x128 = static_cast<uint128_t>(src[i]);
      if (x128 > t_half_) {
        uint64_t u = BarrettReduce(-x128 & mod_t_mask_, mod_qj);
        out[i] = negate_uint_mod(u, mod_qj);
      } else {
        out[i] = BarrettReduce(src[i], mod_qj);
      }
    }
  }

  template <typename Scalar>
  void ModulusDownRNS(absl::Span<const uint64_t> src,
                      absl::Span<Scalar> out) const {
    // Ref: Bajard et al. "A Full RNS Variant of FV like Somewhat Homomorphic
    // Encryption Schemes" (Section 3.2 & 3.3)
    // NOTE(juhou): Basically the same code in seal/util/rns.cpp instead we
    // use the plain modulus `t` as 2^k here.
    using namespace seal::util;
    SPU_ENFORCE_EQ(sizeof(Scalar) * 8, absl::bit_ceil(base_mod_bitlen_),
                   fmt::format("expect base_mod_bitlen={} but got {}",
                               base_mod_bitlen_, sizeof(Scalar) * 8));
    size_t num_modulus = coeff_modulus_size();
    size_t coeff_count = out.size();
    SPU_ENFORCE_EQ(src.size(), num_modulus * out.size());
    SPU_ENFORCE(base_Q_to_gamma_conv_ != nullptr);
    auto cntxt = context_.key_context_data();
    const auto &base_Q = *cntxt->rns_tool()->base_q();
    const auto &coeff_modulus = cntxt->parms().coeff_modulus();
    auto pool = seal::MemoryManager::GetPool();
    auto tmp = allocate_uint(src.size(), pool);

    // 1. multiply with gamma*t
    for (size_t l = 0; l < num_modulus; ++l) {
      const auto *src_ptr = src.data() + l * coeff_count;
      auto *dst_ptr = tmp.get() + l * coeff_count;
      multiply_poly_scalar_coeffmod(src_ptr, coeff_count, gamma_t_mod_Q_[l],
                                    coeff_modulus[l], dst_ptr);
    }

    // 2-1 FastBase convert from baseQ to {gamma}
    auto base_on_gamma = allocate_uint(coeff_count, pool);
    ConstRNSIter inp(tmp.get(), coeff_count);
    RNSIter outp(base_on_gamma.get(), coeff_count);
    base_Q_to_gamma_conv_->fast_convert_array(inp, outp, pool);
    // 2-2 Then multiply with -Q^{-1} mod gamma
    multiply_poly_scalar_coeffmod(base_on_gamma.get(), coeff_count,
                                  neg_inv_Q_mod_gamma_, gamma_,
                                  base_on_gamma.get());

    // 3-1 FastBase convert from baseQ to {t}
    // NOTE: overwrite the `tmp` (tmp is gamma*t*x mod Q)
    const auto *inv_punctured = base_Q.inv_punctured_prod_mod_base_array();
    for (size_t l = 0; l < num_modulus; ++l) {
      auto *src_ptr = tmp.get() + l * coeff_count;
      multiply_poly_scalar_coeffmod(src_ptr, coeff_count, inv_punctured[l],
                                    coeff_modulus[l], src_ptr);
    }

    // sum_i (x * (Q/qi)^{-1} mod qi) * (Q/qi) mod t
    std::vector<Scalar> base_on_t(coeff_count);
    std::fill_n(base_on_t.data(), coeff_count, 0);
    for (size_t l = 0; l < num_modulus; ++l) {
      const Scalar factor = punctured_base_mod_t_[l];
      auto *ptr = tmp.get() + l * coeff_count;
      std::transform(ptr, ptr + coeff_count, base_on_t.data(), base_on_t.data(),
                     [factor](uint64_t x, Scalar y) {
                       return y + static_cast<Scalar>(x) * factor;
                     });
    }

    // 3-2 Then multiply with -Q^{-1} mod t
    std::transform(
        base_on_t.begin(), base_on_t.end(), base_on_t.data(),
        [&](Scalar x) { return (x * neg_inv_Q_mod_t_) & mod_t_mask_; });

    // clang-format off
    // 4 Correct sign: (base_on_t - [base_on_gamma]_gamma) * gamma^{-1} mod t
    // NOTE(juhou):
    // `base_on_gamma` and `base_on_t` together gives
    // `gamma*(x + t*r) + round(gamma*v/q) - e` mod gamma*t for some unknown v and e.
    // (Key point): Taking `base_on_gamma` along equals to
    //    `round(gamma*v/q) - e mod gamma`
    // When gamma > v, e, we can have the centered remainder
    // [round(gamma*v/q) - e mod gamma]_gamma = round(gamma*v/q) - e.
    // As a result, `base_on_t - [base_on_gamma]_gamma mod t` will cancel out the
    // last term and gives `gamma*(x + t*r) mod t`.
    // Finally, multiply with `gamma^{-1} mod t` gives `x mod t`.
    // clang-format on
    uint64_t gamma_div_2 = gamma_.value() >> 1;
    std::transform(
        base_on_gamma.get(), base_on_gamma.get() + coeff_count,
        base_on_t.data(), out.data(), [&](uint64_t on_gamma, Scalar on_t) {
          // [0, gamma) -> [-gamma/2, gamma/2]
          if (on_gamma > gamma_div_2) {
            return ((on_t + gamma_.value() - on_gamma) * inv_gamma_mod_t_) &
                   mod_t_mask_;
          } else {
            return ((on_t - on_gamma) * inv_gamma_mod_t_) & mod_t_mask_;
          }
        });
  }

 private:
  void Init();

  uint32_t base_mod_bitlen_;

  seal::Modulus gamma_;
  uint128_t neg_inv_Q_mod_t_;  // -Q^{-1} mod t
  uint128_t inv_gamma_mod_t_;  // gamma^{-1} mod t

  seal::util::MultiplyUIntModOperand neg_inv_Q_mod_gamma_;  // -Q^{-1} mod gamma
  std::vector<uint128_t> punctured_base_mod_t_;
  std::vector<seal::util::MultiplyUIntModOperand> gamma_t_mod_Q_;
  std::shared_ptr<seal::util::BaseConverter> base_Q_to_gamma_conv_;

  uint128_t mod_t_mask_;
  uint128_t t_half_;
  uint128_t Q_mod_t_;
  std::vector<seal::util::MultiplyUIntModOperand> Q_div_t_mod_qi_;

  seal::SEALContext context_;
};

void ModulusSwitchHelper::Impl::Init() {
  auto cntxt_dat = context_.key_context_data();
  uint32_t logQ = cntxt_dat->total_coeff_modulus_bit_count();

  // Use aux base `gamma` for modulus-down
  size_t poly_degree = cntxt_dat->parms().poly_modulus_degree();
  const auto &coeff_modulus = cntxt_dat->parms().coeff_modulus();
  size_t num_modulus = coeff_modulus.size();
  const int gamma_bits = SEAL_INTERNAL_MOD_BIT_COUNT;
  gamma_ = seal::util::get_primes(poly_degree, gamma_bits, 1)[0];
  for (const auto &modulus : coeff_modulus) {
    SPU_ENFORCE(gamma_.value() != modulus.value(), "Use smaller coeff_modulus");
  }

  if (base_mod_bitlen_ == 128) {
    mod_t_mask_ = static_cast<uint128_t>(-1);
  } else {
    mod_t_mask_ = (static_cast<uint128_t>(1) << base_mod_bitlen_) - 1;
  }
  t_half_ = static_cast<uint128_t>(1ULL) << (base_mod_bitlen_ - 1);

  const auto *bigQ = cntxt_dat->total_coeff_modulus();
  std::vector<uint64_t> Q_div_t(num_modulus, 0);  // Q div t

  // Q div t for t = 2^{base_mod_bitlen_}
  if (logQ > base_mod_bitlen_) {
    seal::util::right_shift_uint(bigQ, base_mod_bitlen_, num_modulus,
                                 Q_div_t.data());
  } else {
    std::fill_n(Q_div_t.data(), num_modulus, 0);
  }

  // Q mod t for t = 2^{base_mod_bitlen_}
  Q_mod_t_ = ModLimbsRing2k(bigQ, num_modulus, base_mod_bitlen_);

  // convert position form to RNS form
  auto pool = seal::MemoryManager::GetPool();
  const auto *rns_tool = cntxt_dat->rns_tool();
  rns_tool->base_q()->decompose(Q_div_t.data(), pool);
  Q_div_t_mod_qi_.resize(num_modulus);
  for (size_t i = 0; i < num_modulus; ++i) {
    Q_div_t_mod_qi_[i].set(Q_div_t[i], coeff_modulus[i]);
  }

  const auto &base_Q = *cntxt_dat->rns_tool()->base_q();
  base_Q_to_gamma_conv_ = std::make_shared<seal::util::BaseConverter>(
      base_Q, seal::util::RNSBase({gamma_}, pool), pool);

  // Q/qi mod t
  punctured_base_mod_t_.resize(num_modulus);
  for (size_t l = 0; l < num_modulus; ++l) {
    const uint64_t *Q_qi_limbs =
        base_Q.punctured_prod_array() + l * num_modulus;
    punctured_base_mod_t_[l] =
        ModLimbsRing2k(Q_qi_limbs, num_modulus, base_mod_bitlen_);
  }

  // -Q^{-1} mod t
  // gamma^{-1} mod t
  if (base_mod_bitlen_ <= 64) {
    neg_inv_Q_mod_t_ = (-Inv2k(base_Q.base_prod()[0])) & mod_t_mask_;
    inv_gamma_mod_t_ = Inv2k(gamma_.value()) & mod_t_mask_;
  } else {
    uint128_t base_Q_128 = AssignU128(
        base_Q.base_prod()[0], num_modulus > 1 ? base_Q.base_prod()[1] : 0);
    uint128_t gamma_128 = AssignU128(gamma_.value(), 0);

    neg_inv_Q_mod_t_ = (-Inv2k(base_Q_128)) & mod_t_mask_;
    inv_gamma_mod_t_ = Inv2k(gamma_128) & mod_t_mask_;

    SPU_ENFORCE_EQ((-neg_inv_Q_mod_t_ * base_Q_128) & mod_t_mask_,
                   static_cast<uint128_t>(1));
    SPU_ENFORCE_EQ((inv_gamma_mod_t_ * gamma_128) & mod_t_mask_,
                   static_cast<uint128_t>(1));
  }

  // -Q^{-1} mod gamma
  neg_inv_Q_mod_gamma_ = [&]() {
    using namespace seal::util;
    auto Q_mod_gamma = modulo_uint(base_Q.base_prod(), num_modulus, gamma_);
    uint64_t inv;
    SPU_ENFORCE(try_invert_uint_mod(Q_mod_gamma, gamma_, inv));
    MultiplyUIntModOperand ret;
    ret.set(negate_uint_mod(inv, gamma_), gamma_);
    return ret;
  }();

  // gamma*t mod Q
  gamma_t_mod_Q_.resize(num_modulus);
  uint128_t t0 = static_cast<uint128_t>(1) << (base_mod_bitlen_ / 2);
  uint128_t t1 = static_cast<uint128_t>(1)
                 << (base_mod_bitlen_ - base_mod_bitlen_ / 2);

  std::transform(coeff_modulus.begin(), coeff_modulus.end(),
                 gamma_t_mod_Q_.data(), [&](const seal::Modulus &prime) {
                   using namespace seal::util;
                   // 2^k0 mod prime
                   uint64_t t = BarrettReduce(t0, prime);
                   // 2^k0 mod prime * 2^k1 mod prime -> 2^k mod prime
                   t = multiply_uint_mod(t, BarrettReduce(t1, prime), prime);

                   MultiplyUIntModOperand ret;
                   uint64_t g = barrett_reduce_64(gamma_.value(), prime);
                   ret.set(multiply_uint_mod(g, t, prime), prime);
                   return ret;
                 });
}

ModulusSwitchHelper::ModulusSwitchHelper(const seal::SEALContext &seal_context,
                                         uint32_t base_mod_bitlen) {
  impl_ = std::make_shared<Impl>(base_mod_bitlen, seal_context);
}

#define DEFINE_MODSWITCH_FUNS(Scalar)                                      \
  void ModulusSwitchHelper::ModulusDownRNS(absl::Span<const uint64_t> src, \
                                           absl::Span<Scalar> out) const { \
    yacl::CheckNotNull(impl_.get());                                       \
    impl_->ModulusDownRNS(src, out);                                       \
  }

DEFINE_MODSWITCH_FUNS(uint32_t)
DEFINE_MODSWITCH_FUNS(uint64_t)
DEFINE_MODSWITCH_FUNS(uint128_t)

#undef DEFINE_MODSWITCH_FUNS

void ModulusSwitchHelper::ModulusUpAt(const ArrayRef &src, size_t mod_idx,
                                      absl::Span<uint64_t> out) const {
  yacl::CheckNotNull(impl_.get());
  const Type &eltype = src.eltype();
  const size_t numel = src.numel();
  SPU_ENFORCE_EQ(numel, out.size());
  SPU_ENFORCE(eltype.isa<RingTy>(), "source must be ring_type, got={}", eltype);
  const auto field = eltype.as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(field, "ModulusUpAt", [&]() {
    using ring2u = std::make_unsigned<ring2k_t>::type;
    impl_->ModulusUpAt(ArrayView<ring2u>(src), mod_idx, out);
  });
}

void ModulusSwitchHelper::CenteralizeAt(const ArrayRef &src, size_t mod_idx,
                                        absl::Span<uint64_t> out) const {
  yacl::CheckNotNull(impl_.get());
  const Type &eltype = src.eltype();
  const size_t numel = src.numel();
  SPU_ENFORCE_EQ(numel, out.size());
  SPU_ENFORCE(eltype.isa<RingTy>(), "source must be ring_type, got={}", eltype);
  const auto field = eltype.as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using ring2u = std::make_unsigned<ring2k_t>::type;
    impl_->CenteralizeAt(ArrayView<ring2u>(src), mod_idx, out);
  });
}

ArrayRef ModulusSwitchHelper::ModulusDownRNS(
    FieldType field, absl::Span<const uint64_t> src) const {
  yacl::CheckNotNull(impl_.get());
  size_t num_modulus = impl_->coeff_modulus_size();
  size_t num_elt = src.size() / num_modulus;
  SPU_ENFORCE_EQ(num_elt * num_modulus, src.size());

  auto out = ring_zeros(field, num_elt);
  ModulusDownRNS(src, out);
  return out;
}

void ModulusSwitchHelper::ModulusDownRNS(absl::Span<const uint64_t> src,
                                         ArrayRef out) const {
  yacl::CheckNotNull(impl_.get());
  auto eltype = out.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  auto field = eltype.as<Ring2k>()->field();
  SPU_ENFORCE(out.isCompact());

  size_t num_modulus = impl_->coeff_modulus_size();
  size_t num_elt = out.numel();
  SPU_ENFORCE_EQ(num_elt * num_modulus, src.size());

  return DISPATCH_ALL_FIELDS(field, "", [&]() {
    using ring2u = std::make_unsigned<ring2k_t>::type;
    absl::Span<ring2u> out_wrap(reinterpret_cast<ring2u *>(out.data()),
                                num_elt);
    impl_->ModulusDownRNS(src, out_wrap);
  });
}

seal::parms_id_type ModulusSwitchHelper::parms_id() const {
  yacl::CheckNotNull(impl_.get());
  return impl_->parms_id();
}

uint32_t ModulusSwitchHelper::base_mod_bitlen() const {
  yacl::CheckNotNull(impl_.get());
  return impl_->base_mod_bitlen();
}

uint32_t ModulusSwitchHelper::coeff_modulus_size() const {
  yacl::CheckNotNull(impl_.get());
  return impl_->coeff_modulus_size();
}

}  // namespace spu::mpc::cheetah
