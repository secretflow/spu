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
#include "libspu/mpc/cheetah/rlwe/packlwes.h"

#include <deque>

#include "seal/ciphertext.h"
#include "seal/evaluator.h"
#include "seal/galoiskeys.h"
#include "seal/keygenerator.h"
#include "seal/plaintext.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/valcheck.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/rlwe/lwe_ct.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

namespace spu::mpc::cheetah {

static void NegacyclicRightShiftInplace(RLWECt &ct, size_t shift,
                                        const seal::SEALContext &context);

static size_t calculateWorkLoad(size_t num_jobs, size_t num_cores = 0) {
  if (num_cores == 0) {
    num_cores = spu::getNumberOfProc();
  }
  return (num_jobs + num_cores - 1) / num_cores;
}

PackingHelper::PackingHelper(size_t gap, const seal::GaloisKeys &galois_keys,
                             const seal::SEALContext &gk_context,
                             const seal::SEALContext &context)
    : gap_(gap),
      galois_keys_(galois_keys),
      gk_context_(gk_context),
      context_(context) {
  SPU_ENFORCE(gk_context_.parameters_set());
  SPU_ENFORCE(seal::is_metadata_valid_for(galois_keys, gk_context));
  SPU_ENFORCE(context_.parameters_set());
  SPU_ENFORCE(gap > 0 && absl::has_single_bit(gap), "invalid gap={}", gap);

  // NOTE(lwj): dirty hack on SEAL's parms_id
  if (context.key_parms_id() != gk_context_.key_parms_id()) {
    SPU_ENFORCE_GT(context_.first_context_data()->chain_index(),
                   gk_context_.first_context_data()->chain_index());
  }

  auto n = gk_context.key_context_data()->parms().poly_modulus_degree();

  size_t ks_level = absl::bit_width(gap) - 1;
  for (size_t i = 0; i < ks_level; ++i) {
    uint32_t galois = (n / (1 << i)) + 1;
    SPU_ENFORCE(galois_keys.has_key(galois), "missing galois={}", galois);
  }

  // pre-compute gap^{-1} mod Q
  auto cntxt = context_.first_context_data();
  const auto &modulus = cntxt->parms().coeff_modulus();
  size_t num_modulus = modulus.size();
  inv_gap_.resize(num_modulus);
  for (size_t i = 0; i < modulus.size(); ++i) {
    uint64_t s = seal::util::barrett_reduce_64(gap, modulus[i]);
    uint64_t _inv;
    SPU_ENFORCE(seal::util::try_invert_uint_mod(s, modulus[i], _inv),
                "failed to compute {}^{-1} mod {}", gap, modulus[i].value());
    inv_gap_[i].set(_inv, modulus[i]);
  }
}

void PackingHelper::MultiplyFixedScalarInplace(RLWECt &ct) const {
  auto cntxt = context_.get_context_data(ct.parms_id());
  SPU_ENFORCE(cntxt != nullptr, "invalid ct");
  const auto &modulus = cntxt->parms().coeff_modulus();
  size_t num_modulus = ct.coeff_modulus_size();
  size_t num_coeff = ct.poly_modulus_degree();
  SPU_ENFORCE(num_modulus <= inv_gap_.size(), "invalid ct");

  for (size_t k = 0; k < ct.size(); ++k) {
    uint64_t *dst_ptr = ct.data(k);
    for (size_t l = 0; l < num_modulus; ++l) {
      seal::util::multiply_poly_scalar_coeffmod(dst_ptr, num_coeff, inv_gap_[l],
                                                modulus.at(l), dst_ptr);
      dst_ptr += num_coeff;
    }
  }
}

void PackingHelper::PackingWithModulusDrop(absl::Span<RLWECt> rlwes,
                                           RLWECt &packed) const {
  if (rlwes.empty()) {
    packed.release();
    return;
  }
  SPU_ENFORCE(rlwes.size() <= gap_);

  auto pid = rlwes[0].parms_id();
  for (auto &rlwe : rlwes) {
    if (rlwe.size() == 0) {
      continue;
    }
    SPU_ENFORCE(rlwe.size() == 2);
    SPU_ENFORCE(pid == rlwe.parms_id());
  }

  doPackingRLWEs(rlwes, packed);
}

void PackingHelper::doPackingRLWEs(absl::Span<RLWECt> rlwes,
                                   RLWECt &out) const {
  auto cntxt = context_.first_context_data();

  int64_t poly_degree = cntxt->parms().poly_modulus_degree();
  int64_t num_ct = rlwes.size();

  SPU_ENFORCE(num_ct > 0 && num_ct <= (int)gap_,
              fmt::format("invalid #rlwes = {} for gap = {}", num_ct, gap_));

  size_t modulus_for_keyswitch =
      gk_context_.first_context_data()->chain_index() + 1;

  yacl::parallel_for(
      0, num_ct, calculateWorkLoad(num_ct), [&](int64_t bgn, int64_t end) {
        for (int64_t i = bgn; i < end; ++i) {
          InvNttInplace(rlwes[i], context_, true);
          // multiply gap^{-1} mod Q
          MultiplyFixedScalarInplace(rlwes[i]);
          // drop some modulus aiming a lighter KeySwitch
          ModulusSwtichInplace(rlwes[i], modulus_for_keyswitch, context_);
          // change pid to galois_context for KS
          rlwes[i].parms_id() = gk_context_.first_parms_id();
        }
      });

  // FFT-like method to merge RLWEs into one RLWE.
  seal::Evaluator evaluator(gk_context_);
  const int64_t logn = absl::bit_width(gap_) - 1;
  for (int64_t k = logn; k >= 1; --k) {
    int64_t h = 1 << (k - 1);
    yacl::parallel_for(
        0, h, calculateWorkLoad(h), [&](int64_t bgn, int64_t end) {
          RLWECt dummy;  // zero-padding with zero RLWE
          for (int64_t i = bgn; i < end; ++i) {
            // E' <- E + X^k*O + Auto(E - X^k*O, k')
            RLWECt &ct_even = i < num_ct ? rlwes[i] : dummy;
            RLWECt &ct_odd = i + h < num_ct ? rlwes[i + h] : dummy;

            bool is_odd_empty = ct_odd.size() == 0;
            bool is_even_empty = ct_even.size() == 0;
            if (is_even_empty && is_odd_empty) {
              ct_even.release();
              continue;
            }

            NegacyclicRightShiftInplace(ct_odd, h, gk_context_);

            if (!is_even_empty) {
              seal::Ciphertext tmp = ct_even;
              if (!is_odd_empty) {
                // E - X^k*O
                // E + X^k*O
                CATCH_SEAL_ERROR(evaluator.sub_inplace(ct_even, ct_odd));
                CATCH_SEAL_ERROR(evaluator.add_inplace(tmp, ct_odd));
              }

              CATCH_SEAL_ERROR(evaluator.apply_galois_inplace(
                  ct_even, poly_degree / h + 1, galois_keys_));
              CATCH_SEAL_ERROR(evaluator.add_inplace(ct_even, tmp));
            } else {
              evaluator.negate(ct_odd, ct_even);
              CATCH_SEAL_ERROR(evaluator.apply_galois_inplace(
                  ct_even, poly_degree / h + 1, galois_keys_));
              CATCH_SEAL_ERROR(evaluator.add_inplace(ct_even, ct_odd));
            }
          }
        });
  }

  SPU_ENFORCE(rlwes[0].size() > 0, fmt::format("all empty RLWEs are invalid"));
  out = rlwes[0];

  out.parms_id() = [&]() -> seal::parms_id_type {
    auto cntxt = context_.first_context_data();
    while ((cntxt->chain_index() + 1) > modulus_for_keyswitch) {
      cntxt = cntxt->next_context_data();
    }
    return cntxt->parms_id();
  }();
}

void GenerateGaloisKeyForPacking(const seal::SEALContext &context,
                                 const RLWESecretKey &key, bool save_seed,
                                 GaloisKeys *out) {
  SPU_ENFORCE(out != nullptr);
  SPU_ENFORCE(context.parameters_set());
  SPU_ENFORCE(seal::is_metadata_valid_for(key, context));

  size_t N = context.key_context_data()->parms().poly_modulus_degree();
  size_t logN = absl::bit_width(N) - 1;
  std::vector<uint32_t> galois_elt;
  for (uint32_t i = 1; i <= logN; i++) {
    galois_elt.push_back((1u << i) + 1);
  }

  seal::KeyGenerator keygen(context, key);

  if (save_seed) {
    auto gk = keygen.create_galois_keys(galois_elt);
    *out = gk.obj();
  } else {
    keygen.create_galois_keys(galois_elt, *out);
  }
}

void NegacyclicRightShiftInplace(RLWECt &ct, size_t shift,
                                 const seal::SEALContext &context) {
  if (shift == 0 || ct.size() == 0) {
    // nothing to do
    return;
  }

  auto cntxt = context.get_context_data(ct.parms_id());
  SPU_ENFORCE(cntxt != nullptr, "invalid ct");
  SPU_ENFORCE(not ct.is_ntt_form(), "need non-ntt ct for negacyclic shift");

  size_t num_coeff = ct.poly_modulus_degree();
  SPU_ENFORCE(shift < num_coeff);

  std::vector<uint64_t> tmp(shift);
  //  i < N - s  ai*X^i -> ai*X^{i + s}
  // i >= N - s ai*X^i -> -ai*X^{(i + s) % N}
  const auto &modulus = cntxt->parms().coeff_modulus();
  for (size_t k = 0; k < ct.size(); ++k) {
    uint64_t *dst_ptr = ct.data(k);

    for (const auto &prime : modulus) {
      // save [N-s, N)
      std::copy_n(dst_ptr + num_coeff - shift, shift, tmp.data());

      // X^i for i \in [0, N-s)
      for (size_t i = num_coeff - shift; i > 0; --i) {
        dst_ptr[i - 1 + shift] = dst_ptr[i - 1];
      }

      // i \n [N-s, N)
      for (size_t i = 0; i < shift; ++i) {
        dst_ptr[i] = seal::util::negate_uint_mod(tmp[i], prime);
      }

      dst_ptr += num_coeff;
    }
  }
}

static void doPackingLWEs(absl::Span<RLWECt> rlwes, const GaloisKeys &galois,
                          const seal::SEALContext &context, RLWECt *out,
                          bool apply_trace = false) {
  SPU_ENFORCE(out != nullptr);
  SPU_ENFORCE(context.parameters_set());
  SPU_ENFORCE(seal::is_metadata_valid_for(galois, context));

  auto cntxt = context.first_context_data();

  size_t poly_degree = cntxt->parms().poly_modulus_degree();
  size_t num_ct = rlwes.size();

  SPU_ENFORCE(
      num_ct <= poly_degree && absl::has_single_bit(num_ct),
      fmt::format("invalid #rlwes = {} for degree = {}", num_ct, poly_degree));

  // FFT-like method to merge RLWEs into one RLWE.
  seal::Evaluator evaluator(context);
  size_t depth = 1;
  while (depth <= num_ct) {
    size_t n = num_ct / depth;
    size_t h = n / 2;
    depth <<= 1;

    auto merge_callback = [&](int64_t start, int64_t end) {
      using namespace seal::util;
      for (int64_t i = start; i < end; ++i) {
        RLWECt &ct_even = rlwes[i];
        RLWECt &ct_odd = rlwes[i + h];

        bool is_odd_empty = ct_odd.size() == 0;
        bool is_even_empty = ct_even.size() == 0;
        if (is_even_empty && is_odd_empty) {
          ct_even.release();
          continue;
        }

        // GS-style butterfly
        // E' <- E + X^k*O + Auto(E - X^k*O, k')
        // O' <- E + X^k*O + Auto(E + X^k*O, k')
        if (!is_odd_empty) {
          NegacyclicRightShiftInplace(ct_odd, poly_degree / depth, context);
        }

        if (!is_even_empty) {
          RLWECt tmp = ct_even;
          if (!is_odd_empty) {
            CATCH_SEAL_ERROR(evaluator.sub_inplace(ct_even, ct_odd));
            CATCH_SEAL_ERROR(evaluator.add_inplace(tmp, ct_odd));
          }
          CATCH_SEAL_ERROR(evaluator.apply_galois_inplace(
              ct_even, static_cast<uint32_t>(depth + 1), galois));
          CATCH_SEAL_ERROR(evaluator.add_inplace(ct_even, tmp));
        } else {
          evaluator.negate(ct_odd, ct_even);
          CATCH_SEAL_ERROR(evaluator.apply_galois_inplace(
              ct_even, static_cast<uint32_t>(depth + 1), galois));
          CATCH_SEAL_ERROR(evaluator.add_inplace(ct_even, ct_odd));
        }
      }
    };

    if (h > 0) {
      yacl::parallel_for(0, h, calculateWorkLoad(h), merge_callback);
    }
  }

  SPU_ENFORCE(rlwes[0].size() > 0, fmt::format("all empty LWes are invalid"));
  *out = rlwes[0];
  out->is_ntt_form() = false;
  out->scale() = 1.;
  if (not apply_trace) {
    return;
  }

  // Step 2 to remove the extra factor (i.e., N/num_lwes) from Step 2.
  size_t log2N = absl::bit_width(poly_degree) - 1;
  size_t log2Nn = absl::bit_width(poly_degree / num_ct) - 1;

  for (size_t k = 1; k <= log2Nn; ++k) {
    RLWECt tmp{*out};
    uint32_t exp = static_cast<uint32_t>((1UL << (log2N - k + 1)) + 1);
    CATCH_SEAL_ERROR(evaluator.apply_galois_inplace(tmp, exp, galois));
    evaluator.add_inplace(*out, tmp);
  }
}

template <class LWEType>
static void doPackLWEs(absl::Span<const LWEType> lwes, const GaloisKeys &galois,
                       const seal::SEALContext &context, RLWECt *out) {
  SPU_ENFORCE(out != nullptr);
  SPU_ENFORCE(context.parameters_set());
  SPU_ENFORCE(seal::is_metadata_valid_for(galois, context));

  auto cntxt = context.first_context_data();

  size_t poly_degree = cntxt->parms().poly_modulus_degree();
  size_t num_lwes = lwes.size();

  SPU_ENFORCE(
      num_lwes <= poly_degree && absl::has_single_bit(num_lwes),
      fmt::format("invalid #lwes = {} for degree = {}", num_lwes, poly_degree));

  // Step 1: cast all LWEs to RLWEs
  std::vector<RLWECt> rlwes(num_lwes);
  yacl::parallel_for(0, num_lwes, calculateWorkLoad(num_lwes),
                     [&](size_t start, size_t end) {
                       for (size_t i = start; i < end; ++i) {
                         lwes[i].CastAsRLWE(context, poly_degree, &rlwes[i]);
                       }
                     });

  doPackingLWEs(absl::MakeSpan(rlwes), galois, context, out, true);
}

template <class LWEType>
static bool IsValidLWEArray(absl::Span<const LWEType> lwes,
                            size_t *poly_degree) {
  size_t n = lwes.size();
  if (n == 0) {
    return false;
  }

  size_t N = 0;
  seal::parms_id_type pid;

  for (size_t i = 0; i < n; ++i) {
    if (not lwes[i].IsValid()) {
      // skip invalid LWEs
      continue;
    }

    if (N == 0) {
      N = lwes[i].poly_modulus_degree();
      pid = lwes[i].parms_id();
    } else if (lwes[i].poly_modulus_degree() != N ||
               lwes[i].parms_id() != pid) {
      // parms mismatch
      return false;
    }
  }

  // all LWEs are invalid
  if (N == 0 || pid == seal::parms_id_zero) {
    return false;
  }

  if (poly_degree) {
    *poly_degree = N;
  }

  return true;
}

size_t PackLWEs(absl::Span<const LWECt> lwes, const GaloisKeys &galois,
                const seal::SEALContext &context, absl::Span<RLWECt> rlwes) {
  size_t n = lwes.size();
  size_t m = rlwes.size();
  size_t poly_degree{0};

  SPU_ENFORCE(IsValidLWEArray(lwes, &poly_degree),
              "LWE.length mismatch the poly degree");
  size_t out_sze = (n + poly_degree - 1) / poly_degree;

  SPU_ENFORCE(out_sze <= m,
              fmt::format("expect >= {} RLWEs but got={}", out_sze, m));

  for (size_t o = 0; o < out_sze; ++o) {
    size_t bgn = o * poly_degree;
    size_t end = std::min(n, bgn + poly_degree);
    size_t this_batch = end - bgn;

    doPackLWEs(lwes.subspan(bgn, this_batch), galois, context, &rlwes[o]);

    SPU_ENFORCE(not rlwes[o].is_transparent(), "");
  }
  return out_sze;
}

size_t PackLWEs(absl::Span<const PhantomLWECt> lwes, const GaloisKeys &galois,
                const seal::SEALContext &context, absl::Span<RLWECt> rlwes) {
  size_t n = lwes.size();
  size_t m = rlwes.size();
  size_t poly_degree{0};
  SPU_ENFORCE(IsValidLWEArray(lwes, &poly_degree),
              "LWE.length mismatch the poly degree = {}", poly_degree);

  size_t out_sze = (n + poly_degree - 1) / poly_degree;
  SPU_ENFORCE(out_sze <= m,
              fmt::format("expect >= {} RLWEs but got={}", out_sze, m));

  for (size_t o = 0; o < out_sze; ++o) {
    size_t bgn = o * poly_degree;
    size_t end = std::min(n, bgn + poly_degree);
    size_t this_batch = end - bgn;
    doPackLWEs(lwes.subspan(bgn, this_batch), galois, context, &rlwes[o]);
    SPU_ENFORCE(not rlwes[o].is_transparent(), "");
  }
  return out_sze;
}

}  // namespace spu::mpc::cheetah