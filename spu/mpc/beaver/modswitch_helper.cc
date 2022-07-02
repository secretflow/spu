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

// Author: Wen-jie Lu(juhou)

#include "spu/mpc/beaver/modswitch_helper.h"

#include <utility>

#include "yasl/base/int128.h"

#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc {

inline static uint64_t Lo64(uint128_t u) { return static_cast<uint64_t>(u); }

inline static uint64_t Hi64(uint128_t u) {
  return static_cast<uint64_t>(u >> 64);
}

[[maybe_unused]] inline static uint128_t AssignU128(uint64_t u) {
  return static_cast<uint128_t>(u);
}

inline static uint128_t AssignU128(uint64_t lo, uint64_t hi) {
  return (static_cast<uint128_t>(hi) << 64) | lo;
}

struct ModulusSwitchHelper::Impl {
 public:
  explicit Impl(uint32_t base_mod_bitlen, seal::SEALContext context)
      : base_mod_bitlen_(base_mod_bitlen), context_(std::move(context)) {
    YASL_ENFORCE(context_.parameters_set(), "invalid seal context");
    YASL_ENFORCE(base_mod_bitlen <= 64 && base_mod_bitlen >= 2,
                 fmt::format("invalid base bitlen {}", base_mod_bitlen));
    Init();
  }

  template <typename T>
  void ModulusUpAt(absl::Span<const T> src, size_t mod_idx,
                   absl::Span<uint64_t> out) const {
    using namespace seal::util;
    size_t num_modulus = Q_div_t_mod_qi_.size();
    YASL_ENFORCE(mod_idx < num_modulus,
                 fmt::format("ModulusUpAt: invalid mod_idx ({} >= {})", mod_idx,
                             num_modulus));
    auto &modulus = context_.key_context_data()->parms().coeff_modulus();

    auto begin = src.data();
    auto end = begin + src.size();
    // round(Q/t*x) = k*x + round(r*x/t)
    // round(Q/t*x) mod qi = ((k mod qi)*x + round(r*x/t)) mod qi
    std::transform(begin, end, out.data(), [&](T x) {
      // u = (Q mod t)*x mod qi
      auto x64 = static_cast<uint64_t>(x);
      auto u =
          multiply_uint_mod(x64, Q_div_t_mod_qi_[mod_idx], modulus[mod_idx]);
      // v = floor((r*x + t/2)/t) = round(r*x/t)
      uint64_t v = ((Q_mod_t_ * x + t_half_) >> base_mod_bitlen_);
      return barrett_reduce_64(u + v, modulus[mod_idx]);
    });
  }

  template <typename T>
  void CenteralizeAt(absl::Span<const T> src, size_t mod_idx,
                     absl::Span<uint64_t> out) const {
    using namespace seal::util;
    const auto &modulus = context_.key_context_data()->parms().coeff_modulus();
    YASL_ENFORCE(mod_idx < modulus.size(), "Centeralize: invalid mod_idx");
    YASL_ENFORCE(src.size() == out.size(), "Centeralize: size mismatch");

    auto begin = src.data();
    auto end = begin + src.size();
    const auto &mod_qj = modulus[mod_idx];

    // view x \in [0, 2^k) as [-2^{k-1}, 2^{k-1})
    std::transform(begin, end, out.data(), [&](T x) -> uint64_t {
      uint128_t x128 = static_cast<uint128_t>(x);
      if (t_half_ <= x128) {
        uint128_t xneg = t_ - x128;

        auto [high_64, low_64] = yasl::DecomposeUInt128(xneg);
        std::vector<uint64_t> barrett_input{low_64, high_64};
        uint64_t u = barrett_reduce_128(barrett_input.data(), mod_qj);
        return negate_uint_mod(u, mod_qj);
      } else {
        return barrett_reduce_64(static_cast<uint64_t>(x), mod_qj);
      }
    });
  }

  template <typename T>
  void ModulusDownRNS(absl::Span<const uint64_t> src, absl::Span<T> out) const {
    using namespace seal::util;
    size_t num_modulus = Q_div_t_mod_qi_.size();
    YASL_ENFORCE(src.size() == num_modulus * out.size());

    auto pool = seal::MemoryManager::GetPool();
    auto tmp = allocate_uint(src.size(), pool);
    std::copy_n(src.data(), src.size(), tmp.get());

    auto cntxt = context_.key_context_data();
    cntxt->rns_tool()->base_q()->compose_array(tmp.get(), out.size(), pool);

    auto bigQ = cntxt->total_coeff_modulus();
    auto Qhalf = cntxt->upper_half_threshold();
    YASL_ENFORCE(Qhalf != nullptr);

    const uint64_t *bigint_ptr = tmp.get();
    std::vector<uint64_t> prod(num_modulus + 2);
    std::vector<uint64_t> add(num_modulus + 2);
    std::vector<uint64_t> div(num_modulus + 2);

    std::vector<uint64_t> _bigQ(num_modulus + 2, 0);
    std::copy_n((const uint64_t *)bigQ, num_modulus, _bigQ.data());

    for (size_t i = 0; i < out.size(); ++i, bigint_ptr += num_modulus) {
      // x*t
      multiply_uint(bigint_ptr, num_modulus,
                    reinterpret_cast<const uint64_t *>(&t_), 2, num_modulus + 2,
                    prod.data());
      // x*t+(Q/2)
      add_uint(prod.data(), num_modulus + 2, Qhalf, num_modulus, 0, add.size(),
               add.data());
      // floor((x*t+Q/2)/Q)
      divide_uint_inplace(add.data(), _bigQ.data(), num_modulus + 2, div.data(),
                          pool);

      out[i] = static_cast<T>(div[0]);
    }
  }

 private:
  void Init();

  uint32_t base_mod_bitlen_;

  uint128_t t_, t_half_;
  // (Q mod t)
  uint128_t Q_mod_t_;
  // floor(Q div t) mod qi
  std::vector<seal::util::MultiplyUIntModOperand> Q_div_t_mod_qi_;
  // t^-1 mod qi
  std::vector<seal::util::MultiplyUIntModOperand> t_inv_qi_;

  seal::SEALContext context_;
};

void ModulusSwitchHelper::Impl::Init() {
  using namespace seal::util;
  auto cntxt_dat = context_.key_context_data();
  uint32_t logQ = cntxt_dat->total_coeff_modulus_bit_count();
  YASL_ENFORCE(logQ > base_mod_bitlen_,
               fmt::format("logQ <= k:{} <= {}", logQ, base_mod_bitlen_));

  auto &coeff_modulus = cntxt_dat->parms().coeff_modulus();
  size_t num_modulus = coeff_modulus.size();

  t_ = static_cast<uint128_t>(1ULL) << base_mod_bitlen_;
  t_half_ = t_ >> 1;

  auto pool = seal::MemoryManager::GetPool();
  auto bigQ = cntxt_dat->total_coeff_modulus();
  std::vector<uint64_t> temp_t(num_modulus, 0);
  temp_t[0] = Lo64(t_);
  temp_t[1] = Hi64(t_);

  std::vector<uint64_t> Q_div_t(num_modulus);  // Q div t
  std::vector<uint64_t> Q_mod_t(num_modulus);  // Q mod t
  divide_uint(bigQ, temp_t.data(), num_modulus, Q_div_t.data(), Q_mod_t.data(),
              pool);
  // Q_mod t
  Q_mod_t_ = AssignU128(Q_mod_t[0], Q_mod_t[1]);

  // convert position form to RNS form
  auto rns_tool = cntxt_dat->rns_tool();
  rns_tool->base_q()->decompose(Q_div_t.data(), pool);
  Q_div_t_mod_qi_.resize(num_modulus);
  for (size_t i = 0; i < num_modulus; ++i) {
    Q_div_t_mod_qi_[i].set(Q_div_t[i], coeff_modulus[i]);
  }
}

ModulusSwitchHelper::ModulusSwitchHelper(const seal::SEALContext &seal_context,
                                         uint32_t base_mod_bitlen) {
  impl_ = std::make_shared<Impl>(base_mod_bitlen, seal_context);
}

// Given x \in [0, t), compute round(Q/t*x) \in [0, Q)
// Let Q = k*t + r for k := floor(Q/t), r := Q mod t
// round(Q/t*x) = k*x + round(r*x/t)
// round(Q/t*x) mod qi = ((k mod qi)*x + round(r*x/t)) mod qi
void ModulusSwitchHelper::ModulusUpAt(absl::Span<const uint32_t> src,
                                      size_t mod_idx,
                                      absl::Span<uint64_t> out) const {
  yasl::CheckNotNull(impl_.get());
  YASL_ENFORCE(src.size() == out.size());
  return impl_->ModulusUpAt<uint32_t>(src, mod_idx, out);
}

void ModulusSwitchHelper::ModulusUpAt(absl::Span<const uint64_t> src,
                                      size_t mod_idx,
                                      absl::Span<uint64_t> out) const {
  yasl::CheckNotNull(impl_.get());
  YASL_ENFORCE(src.size() == out.size());
  return impl_->ModulusUpAt<uint64_t>(src, mod_idx, out);
}

// Cast [-2^{k-1}, 2^{k-1}) to [0, qj)
void ModulusSwitchHelper::CenteralizeAt(absl::Span<const uint32_t> src,
                                        size_t mod_idx,
                                        absl::Span<uint64_t> out) const {
  yasl::CheckNotNull(impl_.get());
  YASL_ENFORCE(src.size() == out.size());
  return impl_->CenteralizeAt<uint32_t>(src, mod_idx, out);
}

void ModulusSwitchHelper::CenteralizeAt(absl::Span<const uint64_t> src,
                                        size_t mod_idx,
                                        absl::Span<uint64_t> out) const {
  yasl::CheckNotNull(impl_.get());
  YASL_ENFORCE(src.size() == out.size());
  return impl_->CenteralizeAt<uint64_t>(src, mod_idx, out);
}

// out = round(src*t/Q)
void ModulusSwitchHelper::ModulusDownRNS(absl::Span<const uint64_t> src,
                                         absl::Span<uint32_t> out) const {
  yasl::CheckNotNull(impl_.get());
  return impl_->ModulusDownRNS<uint32_t>(src, out);
}

void ModulusSwitchHelper::ModulusDownRNS(absl::Span<const uint64_t> src,
                                         absl::Span<uint64_t> out) const {
  yasl::CheckNotNull(impl_.get());
  return impl_->ModulusDownRNS<uint64_t>(src, out);
}

}  // namespace spu::mpc
