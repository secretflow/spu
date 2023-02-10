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

#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

namespace spu::mpc::cheetah {

static size_t MaximumLazy(const seal::SEALContext &context) {
  if (!context.parameters_set()) return 0;
  auto cntxt_dat = context.first_context_data();
  if (!cntxt_dat) return 0;
  const auto &modulus = cntxt_dat->parms().coeff_modulus();

  constexpr int kBarrettLimit = 62;
  int nbits = 0;
  for (const auto &mod : modulus) {
    nbits = std::max(mod.bit_count(), nbits);
  }
  if (nbits >= kBarrettLimit) return 0;
  return 1UL << std::min(16, kBarrettLimit - nbits);
}

LWECt::LWECt() = default;

LWECt::~LWECt() = default;

LWECt::LWECt(const RLWECt &rlwe, size_t coeff_index,
             const seal::SEALContext &context) {
  SPU_ENFORCE(seal::is_metadata_valid_for(rlwe, context),
              "invalid rlwe ct meta");

  size_t num_coeff = rlwe.poly_modulus_degree();
  size_t num_modulus = rlwe.coeff_modulus_size();

  SPU_ENFORCE(!rlwe.is_ntt_form(), "RLWECt should in non-ntt form");
  SPU_ENFORCE_EQ(rlwe.size(), 2U);
  SPU_ENFORCE(coeff_index < num_coeff,
              fmt::format("coefficient index out-of-bound {} >= {}",
                          coeff_index, num_coeff));

  auto cntxt_dat = context.get_context_data(rlwe.parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr, "invalid RLWECt.parms_id for the context");
  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();

  cnst_term_.resize(num_modulus);
  vec_.resize(num_coeff * num_modulus);
  auto *dst_ptr = vec_.data();
  const auto *src_ptr = rlwe.data(1);
  for (size_t l = 0; l < num_modulus; ++l) {
    // reverse [0, k]
    auto rev_ptr = std::reverse_iterator<uint64_t *>(dst_ptr + coeff_index + 1);
    std::copy_n(src_ptr, coeff_index + 1, rev_ptr);

    // reverse-negate [k + 1, n)
    rev_ptr = std::reverse_iterator<uint64_t *>(dst_ptr + num_coeff);
    std::transform(
        src_ptr + coeff_index + 1, src_ptr + num_coeff, rev_ptr,
        [&](uint64_t u) { return seal::util::negate_uint_mod(u, modulus[l]); });

    cnst_term_[l] = rlwe.data(0)[l * num_coeff + coeff_index];
    src_ptr += num_coeff;
    dst_ptr += num_coeff;
  }

  maximum_lazy_ = MaximumLazy(context);
  lazy_counter_ = 0;
  poly_deg_ = num_coeff;
  vec_.parms_id() = rlwe.parms_id();
  vec_.scale() = rlwe.scale();
}

LWECt &LWECt::NegateInplace(const seal::SEALContext &context) {
  SPU_ENFORCE(IsValid());
  if (lazy_counter_ > 0) {
    Reduce(context);
  }

  auto cntxt_dat = context.get_context_data(parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr);

  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();
  size_t num_coeff = parms.poly_modulus_degree();
  size_t num_modulus = modulus.size();

  auto *op = vec_.data();
  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    negate_poly_coeffmod(op, num_coeff, modulus[l], op);
    cnst_term_[l] = negate_uint_mod(cnst_term_[l], modulus[l]);
    op += num_coeff;
  }
  return *this;
}

LWECt &LWECt::AddInplace(const LWECt &oth, const seal::SEALContext &context) {
  if (!IsValid()) {
    *this = oth;
    return *this;
  }

  if (lazy_counter_ > 0) {
    Reduce(context);
  }
  SPU_ENFORCE(oth.lazy_counter_ == 0, "Call LWECt::Reduce() on RHS");

  SPU_ENFORCE(parms_id() == oth.parms_id());
  auto cntxt_dat = context.get_context_data(parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr);

  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();
  size_t num_coeff = parms.poly_modulus_degree();
  size_t num_modulus = modulus.size();

  auto *op0 = vec_.data();
  const auto *op1 = oth.vec_.data();
  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    add_poly_coeffmod(op0, op1, num_coeff, modulus[l], op0);
    cnst_term_[l] = add_uint_mod(cnst_term_[l], oth.cnst_term_[l], modulus[l]);
    op0 += num_coeff;
    op1 += num_coeff;
  }
  return *this;
}

LWECt &LWECt::SubPlainInplace(const std::vector<uint64_t> &plain,
                              const seal::SEALContext &context) {
  SPU_ENFORCE(IsValid());
  SPU_ENFORCE_EQ(plain.size(), coeff_modulus_size());
  if (lazy_counter_ > 0) {
    Reduce(context);
  }
  SPU_ENFORCE(parms_id() == context.first_parms_id());
  auto cntxt_dat = context.get_context_data(parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr);

  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();
  size_t num_modulus = modulus.size();

  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    SPU_ENFORCE(plain[l] < modulus[l].value());
    cnst_term_[l] = sub_uint_mod(cnst_term_[l], plain[l], modulus[l]);
  }
  return *this;
}

LWECt &LWECt::SubInplace(const LWECt &oth, const seal::SEALContext &context) {
  if (!IsValid()) {
    *this = oth;
    return NegateInplace(context);
  }

  if (lazy_counter_ > 0) {
    Reduce(context);
  }
  SPU_ENFORCE(oth.lazy_counter_ == 0, "Call LWECt::Reduce() on RHS");

  SPU_ENFORCE(parms_id() == oth.parms_id());
  auto cntxt_dat = context.get_context_data(parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr);

  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();
  size_t num_coeff = parms.poly_modulus_degree();
  size_t num_modulus = modulus.size();

  auto *op0 = vec_.data();
  const auto *op1 = oth.vec_.data();
  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    sub_poly_coeffmod(op0, op1, num_coeff, modulus[l], op0);
    cnst_term_[l] = sub_uint_mod(cnst_term_[l], oth.cnst_term_[l], modulus[l]);
    op0 += num_coeff;
    op1 += num_coeff;
  }
  return *this;
}

LWECt &LWECt::AddPlainInplace(const std::vector<uint64_t> &plain,
                              const seal::SEALContext &context) {
  SPU_ENFORCE(IsValid());
  SPU_ENFORCE_EQ(plain.size(), coeff_modulus_size());
  if (lazy_counter_ > 0) {
    Reduce(context);
  }
  SPU_ENFORCE(parms_id() == context.first_parms_id());
  auto cntxt_dat = context.get_context_data(parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr);

  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();
  size_t num_modulus = modulus.size();
  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    SPU_ENFORCE(plain[l] < modulus[l].value());
    cnst_term_[l] = add_uint_mod(cnst_term_[l], plain[l], modulus[l]);
  }
  return *this;
}

LWECt &LWECt::AddLazyInplace(const RLWECt &rlwe, size_t coeff_index,
                             const seal::SEALContext &context) {
  if (!IsValid()) {
    *this = LWECt(rlwe, coeff_index, context);
    return *this;
  }

  if (lazy_counter_ >= maximum_lazy_) {
    Reduce(context);
  }

  size_t num_coeff = poly_modulus_degree();
  size_t num_modulus = coeff_modulus_size();

  SPU_ENFORCE(seal::is_metadata_valid_for(rlwe, context),
              "invalid RLWECt meta");
  SPU_ENFORCE(!rlwe.is_ntt_form(), "RLWECt should in non-ntt form");
  SPU_ENFORCE_EQ(rlwe.size(), 2U);
  SPU_ENFORCE(coeff_index < num_coeff,
              fmt::format("coefficient index out-of-bound {} >= {}",
                          coeff_index, num_coeff));

  SPU_ENFORCE(parms_id() == rlwe.parms_id());

  auto cntxt_dat = context.get_context_data(rlwe.parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr, "invalid RLWECt.parms_id for the context");
  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();

  auto *src_ptr = vec_.data();
  const auto *op_ptr = rlwe.data(1);

  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    // reverse [0, k]
    auto rev_ptr =
        std::reverse_iterator<const uint64_t *>(op_ptr + coeff_index + 1);
    std::transform(src_ptr, src_ptr + coeff_index + 1, rev_ptr, src_ptr,
                   [&](uint64_t u, uint64_t v) { return u + v; });

    // reverse-negate [k + 1, n)
    rev_ptr = std::reverse_iterator<const uint64_t *>(op_ptr + num_coeff);
    std::transform(src_ptr + coeff_index + 1, src_ptr + num_coeff, rev_ptr,
                   src_ptr + coeff_index + 1, [&](uint64_t u, uint64_t v) {
                     return u + modulus[l].value() - v;
                   });
    cnst_term_[l] += rlwe.data(0)[l * num_coeff + coeff_index];

    src_ptr += num_coeff;
    op_ptr += num_coeff;
  }

  lazy_counter_ += 1;
  return *this;
}

LWECt &LWECt::SubLazyInplace(const RLWECt &rlwe, size_t coeff_index,
                             const seal::SEALContext &context) {
  if (!IsValid()) {
    *this = LWECt(rlwe, coeff_index, context);
    return NegateInplace(context);
  }

  if (lazy_counter_ >= maximum_lazy_) {
    Reduce(context);
  }

  size_t num_coeff = poly_modulus_degree();
  size_t num_modulus = coeff_modulus_size();

  SPU_ENFORCE(seal::is_metadata_valid_for(rlwe, context),
              "invalid RLWECt meta");
  SPU_ENFORCE(!rlwe.is_ntt_form(), "RLWECt should in non-ntt form");
  SPU_ENFORCE_EQ(rlwe.size(), 2U);
  SPU_ENFORCE(coeff_index < num_coeff,
              fmt::format("coefficient index out-of-bound {} >= {}",
                          coeff_index, num_coeff));

  SPU_ENFORCE(parms_id() == rlwe.parms_id());

  auto cntxt_dat = context.get_context_data(rlwe.parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr, "invalid RLWECt.parms_id for the context");
  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();

  auto *src_ptr = vec_.data();
  const auto *op_ptr = rlwe.data(1);

  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    // reverse [0, k]
    auto rev_ptr =
        std::reverse_iterator<const uint64_t *>(op_ptr + coeff_index + 1);
    std::transform(
        src_ptr, src_ptr + coeff_index + 1, rev_ptr, src_ptr,
        [&](uint64_t u, uint64_t v) { return u + modulus[l].value() - v; });

    // reverse-negate [k + 1, n)
    rev_ptr = std::reverse_iterator<const uint64_t *>(op_ptr + num_coeff);
    std::transform(src_ptr + coeff_index + 1, src_ptr + num_coeff, rev_ptr,
                   src_ptr + coeff_index + 1,
                   [&](uint64_t u, uint64_t v) { return u + v; });

    cnst_term_[l] +=
        (modulus[l].value() - rlwe.data(0)[l * num_coeff + coeff_index]);

    src_ptr += num_coeff;
    op_ptr += num_coeff;
  }

  lazy_counter_ += 1;
  return *this;
}

void LWECt::Reduce(const seal::SEALContext &context) {
  if (!IsValid()) {
    return;
  }
  if (lazy_counter_ == 0) {
    return;
  }
  SPU_ENFORCE(lazy_counter_ <= maximum_lazy_);

  auto cntxt_dat = context.get_context_data(parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr, "invalid context for this LWECt");

  const auto &parms = cntxt_dat->parms();
  const auto &modulus = parms.coeff_modulus();
  size_t num_coeff = parms.poly_modulus_degree();
  size_t num_modulus = modulus.size();

  SPU_ENFORCE(num_coeff == poly_deg_ && num_modulus == cnst_term_.size(),
              "invalid context for this LWECt");

  auto *src_ptr = vec_.data();
  for (size_t l = 0; l < num_modulus; ++l, src_ptr += num_coeff) {
    seal::util::modulo_poly_coeffs(src_ptr, num_coeff, modulus[l], src_ptr);
    cnst_term_[l] = seal::util::barrett_reduce_64(cnst_term_[l], modulus[l]);
  }

  lazy_counter_ = 0;
}

size_t LWECt::save_size(seal::compr_mode_type compr_mode) const {
  using namespace seal;

  size_t members_size = Serialization::ComprSizeEstimate(
      util::add_safe(
          sizeof(uint32_t),  // num_modulus
          sizeof(uint64_t) * cnst_term_.size(),
          util::safe_cast<size_t>(vec_.save_size(compr_mode_type::none))),
      compr_mode);

  return util::add_safe(sizeof(Serialization::SEALHeader), members_size);
}

void LWECt::save_members(std::ostream &stream) const {
  SPU_ENFORCE(lazy_counter_ == 0, "Call LWECt::Reduce() before saving it");
  auto old_except_mask = stream.exceptions();
  try {
    // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
    stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    vec_.save(stream, seal::compr_mode_type::none);
    auto num_modulus = static_cast<uint32_t>(cnst_term_.size());
    stream.write(reinterpret_cast<const char *>(&num_modulus),
                 sizeof(uint32_t));
    for (uint64_t cnst : cnst_term_) {
      stream.write(reinterpret_cast<const char *>(&cnst), sizeof(uint64_t));
    }
  } catch (const std::ios_base::failure &) {
    stream.exceptions(old_except_mask);
    SPU_THROW("failed to save LWECt due to I/O error");
  } catch (...) {
    stream.exceptions(old_except_mask);
    SPU_THROW("failed to save LWECt");
  }
  stream.exceptions(old_except_mask);
}

void LWECt::load(const seal::SEALContext &context,
                 const seal::seal_byte *buffer, size_t size) {
  LWECt tmp;
  tmp.unsafe_load(context, buffer, size);

  const auto &modulus = context.first_context_data()->parms().coeff_modulus();
  SPU_ENFORCE(coeff_modulus_size() <= modulus.size());
  for (size_t l = 0; l < coeff_modulus_size(); ++l) {
    SPU_ENFORCE(cnst_term_[l] < modulus[l].value());
  }
  SPU_ENFORCE(seal::is_valid_for(tmp.vec_, context));

  std::swap(*this, tmp);
}

void LWECt::load_members(const seal::SEALContext &context, std::istream &stream,
                         SEAL_MAYBE_UNUSED seal::SEALVersion version) {
  SPU_ENFORCE(context.parameters_set());
  auto old_except_mask = stream.exceptions();
  LWECt tmp;
  try {
    // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
    stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

    tmp.vec_.load(context, stream);

    uint32_t num_modulus;
    stream.read(reinterpret_cast<char *>(&num_modulus), sizeof(uint32_t));

    tmp.cnst_term_.resize(num_modulus);
    for (size_t l = 0; l < num_modulus; ++l) {
      stream.read(reinterpret_cast<char *>(&tmp.cnst_term_[l]),
                  sizeof(uint64_t));
    }

    tmp.maximum_lazy_ = MaximumLazy(context);
    tmp.poly_deg_ = tmp.vec_.coeff_count() / num_modulus;
    tmp.lazy_counter_ = 0;
  } catch (const std::ios_base::failure &) {
    stream.exceptions(old_except_mask);
    SPU_THROW("failed to load LWECt due to I/O error");
  } catch (...) {
    stream.exceptions(old_except_mask);
    SPU_THROW("failed to load LWECt");
  }
  stream.exceptions(old_except_mask);

  std::swap(*this, tmp);
}

}  // namespace spu::mpc::cheetah
