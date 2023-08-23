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

#include "libspu/mpc/cheetah/rlwe/utils.h"

#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {

#define ALLOW_BFV_DECRYPTION_FAIL 1

void RemoveCoefficientsInplace(RLWECt& ciphertext,
                               const std::set<size_t>& to_remove) {
  SPU_ENFORCE(!ciphertext.is_ntt_form());
  SPU_ENFORCE_EQ(2UL, ciphertext.size());

  size_t num_to_remove = to_remove.size();
  size_t num_coeff = ciphertext.poly_modulus_degree();
  size_t num_modulus = ciphertext.coeff_modulus_size();
  SPU_ENFORCE(std::all_of(to_remove.begin(), to_remove.end(),
                          [&](size_t idx) { return idx < num_coeff; }));
  SPU_ENFORCE(num_to_remove < num_coeff);
  if (num_to_remove == 0) return;

  for (size_t l = 0; l < num_modulus; ++l) {
    auto ct_ptr = ciphertext.data(0) + l * num_coeff;
    for (size_t idx : to_remove) {
      ct_ptr[idx] = 0;
    }
  }
}

void KeepCoefficientsInplace(RLWECt& ciphertext,
                             const std::set<size_t>& to_keep) {
  SPU_ENFORCE(!ciphertext.is_ntt_form());
  SPU_ENFORCE_EQ(2UL, ciphertext.size());

  size_t num_coeff = ciphertext.poly_modulus_degree();
  SPU_ENFORCE(std::all_of(to_keep.begin(), to_keep.end(),
                          [&](size_t idx) { return idx < num_coeff; }));
  if (to_keep.size() == num_coeff) return;

  std::set<size_t> to_remove;
  for (size_t idx = 0; idx < num_coeff; ++idx) {
    if (to_keep.find(idx) == to_keep.end()) {
      to_remove.insert(idx);
    }
  }
  RemoveCoefficientsInplace(ciphertext, to_remove);
}

// Truncate ciphertexts for a smaller communication.
// NOTE: after truncation, further homomorphic operation is meaningless.
void TruncateBFVForDecryption(seal::Ciphertext& ct,
                              const seal::SEALContext& context) {
  auto make_bits_mask = [](int n_low_zeros) {
    n_low_zeros = std::max(0, n_low_zeros);
    n_low_zeros = std::min(63, n_low_zeros);
    return (static_cast<uint64_t>(-1) >> n_low_zeros) << n_low_zeros;
  };

  auto context_data = context.last_context_data();

  const auto& parms = context_data->parms();
  SPU_ENFORCE(parms.scheme() == seal::scheme_type::bfv,
              "TruncateSEALCtInplace: scheme_type not supported");
  SPU_ENFORCE(ct.size() == 2, "TruncateSEALCtInplace: invalid ct.size");
  SPU_ENFORCE(ct.coeff_modulus_size() == 1,
              "TruncateSEALCtInplace: invalid ct.coeff_modulus_size");
  SPU_ENFORCE(!ct.is_ntt_form(),
              "TruncateSEALCtInplace: invalid ct.is_ntt_form");
  // Hack on BFV decryption formula: c0 + c1*s mod p0 = m' = Delta*m + e ->
  // round(m'/Delta) = m The low-end bits of c0, c1 are useless for decryption,
  // and thus we can trucate those bits
  const size_t poly_n = ct.poly_modulus_degree();
  // Delta := round(p0/t), Then floor(log2(Delta)) = ceil(log2(p0)) -
  // ceil(log2(t)) - 1
  const int n_delta_bits =
      parms.coeff_modulus()[0].bit_count() - parms.plain_modulus().bit_count();
  const uint64_t mask0 = make_bits_mask(n_delta_bits - 2);
  std::transform(ct.data(0), ct.data(0) + poly_n, ct.data(0),
                 [mask0](uint64_t u) { return u & mask0; });

  // NOTE(juhou): The following truncation might lead to a decryption failure.
  // For for error sensitive computation (eg Beaver's triples), we just skip
  // this truncation.
#if ALLOW_BFV_DECRYPTION_FAIL
  // Norm |c1 * s|_infty < |c1|_infty * |s|_infty.
  // The value of |c1|_infty * |s|_infty is heuristically bounded by 12. *
  // Std(|c1|_infty) * Std(|s|_infty) Assume |c1| < B is B-bounded uniform. Then
  // the variance Var(|c1|_infty) = B^2*N/12. We need to make sure the value |c1
  // * s|_infty is NOT overflow Delta.
  constexpr double heuristic_bound = 12.;  // P(|x| > Delta) < 2^−{40}
  int n_var_bits{0};
  // The variance Var(|s|_infty) = 2/3*N since the secret key s is uniform
  // from {-1, 0, 1}.
  n_var_bits = std::log2(heuristic_bound * poly_n * std::sqrt(1 / 18.));
  const uint64_t mask1 = make_bits_mask(n_delta_bits - n_var_bits);
  std::transform(ct.data(1), ct.data(1) + poly_n, ct.data(1),
                 [mask1](uint64_t u) { return u & mask1; });
#endif
}

void NttInplace(RLWECt& ct, const seal::SEALContext& context, bool lazy) {
  namespace sut = seal::util;
  if (ct.is_ntt_form()) {
    return;
  }

  SPU_ENFORCE(context.parameters_set());
  auto cntxt_data = context.get_context_data(ct.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);

  const auto* ntt_tables = cntxt_data->small_ntt_tables();

  for (size_t k = 0; k < ct.size(); ++k) {
    if (lazy) {
      sut::ntt_negacyclic_harvey_lazy(sut::iter(ct)[k], ct.coeff_modulus_size(),
                                      ntt_tables);
    } else {
      sut::ntt_negacyclic_harvey(sut::iter(ct)[k], ct.coeff_modulus_size(),
                                 ntt_tables);
    }
  }
  ct.is_ntt_form() = true;
}

void InvNttInplace(RLWECt& ct, const seal::SEALContext& context, bool lazy) {
  namespace sut = seal::util;
  if (not ct.is_ntt_form()) {
    return;
  }

  SPU_ENFORCE(context.parameters_set());
  auto cntxt_data = context.get_context_data(ct.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);

  const auto* ntt_tables = cntxt_data->small_ntt_tables();
  size_t L = ct.coeff_modulus_size();
  for (size_t k = 0; k < ct.size(); ++k) {
    if (lazy) {
      sut::inverse_ntt_negacyclic_harvey_lazy(sut::iter(ct)[k], L, ntt_tables);
    } else {
      sut::inverse_ntt_negacyclic_harvey(sut::iter(ct)[k], L, ntt_tables);
    }
  }
  ct.is_ntt_form() = false;
}

void NttInplace(RLWEPt& pt, const seal::SEALContext& context, bool lazy) {
  SPU_ENFORCE(context.parameters_set());
  auto cntxt_data = context.get_context_data(pt.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);

  auto L = cntxt_data->parms().coeff_modulus().size();
  SPU_ENFORCE(pt.coeff_count() % L == 0);

  const auto* ntt_tables = cntxt_data->small_ntt_tables();
  size_t n = pt.coeff_count() / L;
  auto* pt_ptr = pt.data();
  for (size_t l = 0; l < L; ++l) {
    if (lazy) {
      seal::util::ntt_negacyclic_harvey_lazy(pt_ptr, ntt_tables[l]);
    } else {
      seal::util::ntt_negacyclic_harvey(pt_ptr, ntt_tables[l]);
    }
    pt_ptr += n;
  }
}

void InvNttInplace(RLWEPt& pt, const seal::SEALContext& context, bool lazy) {
  SPU_ENFORCE(context.parameters_set());
  auto cntxt_data = context.get_context_data(pt.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);

  auto L = cntxt_data->parms().coeff_modulus().size();
  SPU_ENFORCE(pt.coeff_count() % L == 0);

  const auto* ntt_tables = cntxt_data->small_ntt_tables();
  size_t n = pt.coeff_count() / L;
  auto* pt_ptr = pt.data();
  for (size_t l = 0; l < L; ++l) {
    if (lazy) {
      seal::util::inverse_ntt_negacyclic_harvey_lazy(pt_ptr, ntt_tables[l]);
    } else {
      seal::util::inverse_ntt_negacyclic_harvey(pt_ptr, ntt_tables[l]);
    }
    pt_ptr += n;
  }
}

// ct <- ct + pt
// Handling ct.parms_id = context.key_context_id
void AddPlainInplace(RLWECt& ct, const RLWEPt& pt,
                     const seal::SEALContext& context) {
  namespace sut = seal::util;
  SPU_ENFORCE(context.parameters_set());
  auto cntxt_data = context.get_context_data(ct.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);
  SPU_ENFORCE(ct.parms_id() == pt.parms_id());

  const auto& modulus = cntxt_data->parms().coeff_modulus();

  sut::RNSIter ct_iter(ct.data(0), ct.poly_modulus_degree());
  sut::ConstRNSIter pt_iter(pt.data(), ct.poly_modulus_degree());

  sut::add_poly_coeffmod(ct_iter, pt_iter, modulus.size(), modulus, ct_iter);
}

// ct <- ct - pt
// Handling ct.parms_id = context.key_context_id
void SubPlainInplace(RLWECt& ct, const RLWEPt& pt,
                     const seal::SEALContext& context) {
  namespace sut = seal::util;
  SPU_ENFORCE(context.parameters_set());
  auto cntxt_data = context.get_context_data(ct.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);
  SPU_ENFORCE(ct.parms_id() == pt.parms_id());

  const auto& modulus = cntxt_data->parms().coeff_modulus();

  sut::RNSIter ct_iter(ct.data(0), ct.poly_modulus_degree());
  sut::ConstRNSIter pt_iter(pt.data(), ct.poly_modulus_degree());

  sut::sub_poly_coeffmod(ct_iter, pt_iter, modulus.size(), modulus, ct_iter);
}

void ModulusSwtichInplace(RLWECt& ct, size_t num_modulus_to_keep,
                          const seal::SEALContext& context) {
  namespace sut = seal::util;
  SPU_ENFORCE(num_modulus_to_keep >= 1 &&
              num_modulus_to_keep <= ct.coeff_modulus_size());
  if (num_modulus_to_keep == ct.coeff_modulus_size()) {
    // nothing to do
    return;
  }

  auto cntxt = context.get_context_data(ct.parms_id());
  YACL_ENFORCE(cntxt != nullptr);
  size_t index = cntxt->chain_index();
  YACL_ENFORCE((index + 1) >= num_modulus_to_keep);

  auto target_context = cntxt;
  auto pool = seal::MemoryManager::GetPool();

  while (target_context->chain_index() >= num_modulus_to_keep) {
    const auto* rns_tool = target_context->rns_tool();
    const auto* ntt_tables = target_context->small_ntt_tables();
    if (ct.is_ntt_form()) {
      SEAL_ITERATE(sut::iter(ct), ct.size(), [&](auto I) {
        rns_tool->divide_and_round_q_last_ntt_inplace(I, ntt_tables, pool);
      });
    } else {
      SEAL_ITERATE(sut::iter(ct), ct.size(), [&](auto I) {
        rns_tool->divide_and_round_q_last_inplace(I, pool);
      });
    }

    auto next_context = target_context->next_context_data();
    SPU_ENFORCE(next_context != nullptr);

    RLWECt next_ct(pool);
    next_ct.resize(context, next_context->parms_id(), ct.size());
    SEAL_ITERATE(sut::iter(ct, next_ct), ct.size(), [&](auto I) {
      sut::set_poly(std::get<0>(I), ct.poly_modulus_degree(),
                    ct.coeff_modulus_size() - 1, std::get<1>(I));
    });
    next_ct.is_ntt_form() = ct.is_ntt_form();
    target_context = next_context;
    std::swap(next_ct, ct);
  }
  SPU_ENFORCE_EQ(num_modulus_to_keep, ct.coeff_modulus_size());
}

// NOTE(lwj): the following code is modified from
// seal/util/rlwe.cpp#encrypt_zero_symmetric
// The symmetric RLWE encryption of m is given as (m + e - a*sk, a)
// The origin SEAL uses a general API encrypt_zero_symmetric() to generate (e
// - a*sk, a) first. Then the encryption of `m` is followed by addition `m + e
// - a*sk`
//
// Observation: we can perform the addition `m + e` first to save one NTT
// That is when `need_ntt=true` NTT(m) + NTT(e) is replaced by NTT(m + e)
void SymmetricRLWEEncrypt(const RLWESecretKey& sk,
                          const seal::SEALContext& context,
                          absl::Span<const RLWEPt> msg_non_ntt, bool need_ntt,
                          bool need_seed, absl::Span<RLWECt> out_ct) {
  using namespace seal;
  using namespace seal::util;
  size_t n = msg_non_ntt.size();
  SPU_ENFORCE(out_ct.size() >= n);
  SPU_ENFORCE(seal::is_metadata_valid_for(sk, context));
  if (n == 0) {
    return;
  }

  SPU_ENFORCE(context.parameters_set(), "invalid SEALContext");
  SPU_ENFORCE(std::all_of(msg_non_ntt.data(), msg_non_ntt.data() + n,
                          [&](const RLWEPt& pt) {
                            return context.get_context_data(pt.parms_id()) !=
                                   nullptr;
                          }),
              "invalid plaintext to encrypt");

  auto pool = MemoryManager::GetPool(mm_prof_opt::mm_force_thread_local, true);
  std::shared_ptr<UniformRandomGenerator> bootstrap_prng = nullptr;

  for (size_t i = 0; i < n; ++i) {
    const RLWEPt& msg = msg_non_ntt[i];
    RLWECt& destination = out_ct[i];

    auto parms_id = msg.parms_id();
    auto& context_data = *context.get_context_data(parms_id);
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();
    auto ntt_tables = context_data.small_ntt_tables();
    size_t encrypted_size = 2;

    // If a polynomial is too small to store UniformRandomGeneratorInfo,
    // it is best to just disable save_seed. Note that the size needed is
    // the size of UniformRandomGeneratorInfo plus one (uint64_t) because
    // of an indicator word that indicates a seeded ciphertext.
    size_t poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);
    if (msg.coeff_count() != poly_uint64_count) {
      throw std::invalid_argument("msg coeff_count mismatch");
    }
    size_t prng_info_byte_count = static_cast<size_t>(
        UniformRandomGeneratorInfo::SaveSize(compr_mode_type::none));
    size_t prng_info_uint64_count = divide_round_up(
        prng_info_byte_count, static_cast<size_t>(bytes_per_uint64));
    if (need_ntt && poly_uint64_count < prng_info_uint64_count + 1) {
      need_ntt = false;
    }

    destination.resize(context, parms_id, encrypted_size);
    destination.is_ntt_form() = need_ntt;
    destination.scale() = 1.0;
    destination.correction_factor() = 1;

    // Create an instance of a random number generator. We use this for
    // sampling a seed for a second PRNG used for sampling u (the seed can be
    // public information. This PRNG is also used for sampling the noise/error
    // below.
    if (!bootstrap_prng) {
      bootstrap_prng = parms.random_generator()->create();
    }

    // Sample a public seed for generating uniform randomness
    prng_seed_type public_prng_seed;
    bootstrap_prng->generate(
        prng_seed_byte_count,
        reinterpret_cast<seal_byte*>(public_prng_seed.data()));

    // Set up a new default PRNG for expanding u from the seed sampled above
    auto ciphertext_prng =
        UniformRandomGeneratorFactory::DefaultFactory()->create(
            public_prng_seed);

    // Generate ciphertext: (c[0], c[1]) = ([msg + e - a*s]_q, a) in BFV/CKKS
    uint64_t* c0 = destination.data();
    uint64_t* c1 = destination.data(1);

    // Sample a uniformly at random
    if (need_ntt || !need_seed) {
      // Sample the NTT form directly
      sample_poly_uniform(ciphertext_prng, parms, c1);
    } else if (need_seed) {
      // Sample non-NTT form and store the seed
      sample_poly_uniform(ciphertext_prng, parms, c1);
      for (size_t i = 0; i < coeff_modulus_size; i++) {
        // Transform the c1 into NTT representation
        ntt_negacyclic_harvey(c1 + i * coeff_count, ntt_tables[i]);
      }
    }

    // Sample e <-- chi
    auto noise(allocate_poly(coeff_count, coeff_modulus_size, pool));
    SEAL_NOISE_SAMPLER(bootstrap_prng, parms, noise.get());

    // Calculate -(as + e) (mod q) and store in c[0] in BFV/CKKS
    for (size_t i = 0; i < coeff_modulus_size; i++) {
      dyadic_product_coeffmod(sk.data().data() + i * coeff_count,
                              c1 + i * coeff_count, coeff_count,
                              coeff_modulus[i], c0 + i * coeff_count);
      if (need_ntt) {
        // Peform the addition m + e first
        // NOTE: lazy reduction here which will be obsorbed by
        // ntt_negacyclic_harvey
        std::transform(noise.get() + i * coeff_count,
                       noise.get() + i * coeff_count + coeff_count,
                       msg.data() + i * coeff_count,
                       noise.get() + i * coeff_count, std::plus<uint64_t>());

        // Then transform m + e to NTT form
        // noise <- m + e
        ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);
      } else {
        // c0 <- a*s - m
        inverse_ntt_negacyclic_harvey(c0 + i * coeff_count, ntt_tables[i]);
        sub_poly_coeffmod(c0 + i * coeff_count, msg.data() + i * coeff_count,
                          coeff_count, coeff_modulus[i], c0 + i * coeff_count);
      }

      // c0 <- noise - c0
      //    <- m + e - a*s   (need_ntt=true)
      //    <- e - (a*s - m) (need_ntt=false)
      sub_poly_coeffmod(noise.get() + i * coeff_count, c0 + i * coeff_count,
                        coeff_count, coeff_modulus[i], c0 + i * coeff_count);
    }

    if (!need_ntt && !need_seed) {
      for (size_t i = 0; i < coeff_modulus_size; i++) {
        // Transform the c1 into non-NTT representation
        inverse_ntt_negacyclic_harvey(c1 + i * coeff_count, ntt_tables[i]);
      }
    }

    if (need_seed) {
      UniformRandomGeneratorInfo prng_info = ciphertext_prng->info();

      // Write prng_info to destination.data(1) after an indicator word
      c1[0] = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);
      prng_info.save(reinterpret_cast<seal_byte*>(c1 + 1), prng_info_byte_count,
                     compr_mode_type::none);
    }
  }
}
}  // namespace spu::mpc::cheetah
