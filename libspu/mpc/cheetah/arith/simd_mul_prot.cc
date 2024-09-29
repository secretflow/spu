// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/arith/simd_mul_prot.h"

#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/scalingvariant.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

namespace spu::mpc::cheetah {

SIMDMulProt::SIMDMulProt(uint64_t simd_lane, uint64_t prime_modulus)
    : simd_lane_(simd_lane), prime_modulus_(prime_modulus) {
  SPU_ENFORCE(prime_modulus_.is_prime(), "modulus {} is not a prime",
              prime_modulus);
  SPU_ENFORCE(absl::has_single_bit(simd_lane), "invalid simd lane {}",
              simd_lane_);
  SPU_ENFORCE(prime_modulus % (2 * simd_lane) == 1);

  encode_tabl_ = std::make_unique<seal::util::NTTTables>(
      absl::bit_width(simd_lane) - 1, prime_modulus_);
}

SIMDMulProt::~SIMDMulProt() = default;

void SIMDMulProt::EncodeBatch(absl::Span<const uint64_t> array,
                              absl::Span<RLWEPt> batch_out) const {
  SPU_ENFORCE_EQ(batch_out.size(), CeilDiv(array.size(), (size_t)simd_lane_));

  for (size_t i = 0; i < batch_out.size(); ++i) {
    int64_t slice_bgn = i * simd_lane_;
    int64_t slice_n = std::min<int64_t>(simd_lane_, array.size() - slice_bgn);

    EncodeSingle(array.subspan(slice_bgn, slice_n), batch_out[i]);
  }
}

void SIMDMulProt::EncodeSingle(absl::Span<const uint64_t> array,
                               RLWEPt &out) const {
  SPU_ENFORCE_LE(array.size(), (size_t)simd_lane_);

  SPU_ENFORCE(
      std::all_of(array.cbegin(), array.cend(),
                  [&](uint64_t x) { return x < prime_modulus_.value(); }),
      "array value out-of-range to encode");

  out.parms_id() = seal::parms_id_zero;
  out.resize(simd_lane_);
  std::copy_n(array.data(), array.size(), out.data());
  std::fill_n(out.data() + array.size(), simd_lane_ - array.size(), 0);

  // SIMD encode is doing intt
  // NOTE(lwj): we do not need rotation, and thus we are skipping the position
  // arrangement in seal::BatchEncoder
  seal::util::inverse_ntt_negacyclic_harvey(out.data(), *encode_tabl_);
}

void SIMDMulProt::DecodeSingle(const RLWEPt &poly,
                               absl::Span<uint64_t> array) const {
  SPU_ENFORCE_EQ(poly.coeff_count(), (size_t)simd_lane_);
  SPU_ENFORCE_LE(array.size(), poly.coeff_count());
  if (array.empty()) {
    return;
  }

  if (array.size() == (size_t)simd_lane_) {
    // SIMD encode is doing intt
    // inplace ntt
    std::copy_n(poly.data(), simd_lane_, array.data());
    seal::util::ntt_negacyclic_harvey(array.data(), *encode_tabl_);
  } else {
    // only take the front part
    std::vector<uint64_t> tmp(simd_lane_);
    std::copy_n(poly.data(), simd_lane_, tmp.data());
    seal::util::ntt_negacyclic_harvey(tmp.data(), *encode_tabl_);
    std::copy_n(tmp.data(), array.size(), array.data());
  }
}

void SIMDMulProt::SymEncrypt(absl::Span<const RLWEPt> polys,
                             const RLWESecretKey &secret_key,
                             const seal::SEALContext &context, bool save_seed,
                             absl::Span<RLWECt> out) const {
  SPU_ENFORCE_EQ(polys.size(), out.size());
  for (size_t i = 0; i < polys.size(); ++i) {
    seal::util::encrypt_zero_symmetric(secret_key, context,
                                       context.first_parms_id(), false,
                                       save_seed, out[i]);
    seal::util::multiply_add_plain_with_scaling_variant(
        polys[i], *context.first_context_data(),
        seal::util::RNSIter{out[i].data(), out[i].poly_modulus_degree()});
  }
}

// Compute ct * pt - mask mod p
void SIMDMulProt::MulThenReshareInplaceOneBit(
    absl::Span<RLWECt> ct, absl::Span<const RLWEPt> pt,
    absl::Span<uint64_t> share_mask, const RLWEPublicKey &public_key,
    const seal::SEALContext &context) {
  SPU_ENFORCE_EQ(ct.size(), pt.size());
  SPU_ENFORCE_EQ(CeilDiv(share_mask.size(), (size_t)simd_lane_), pt.size());

  seal::Evaluator evaluator(context);
  RLWECt zero_enc;

  constexpr int kMarginBitsForDec = 10;
  seal::parms_id_type final_level_id = context.last_parms_id();
  while (final_level_id != context.first_parms_id()) {
    auto cntxt = context.get_context_data(final_level_id);
    if (cntxt->total_coeff_modulus_bit_count() >=
        kMarginBitsForDec + cntxt->parms().plain_modulus().bit_count()) {
      break;
    }
    final_level_id = cntxt->prev_context_data()->parms_id();
  }

  RLWEPt rnd;
  RLWEPt tmp_poly;
  tmp_poly.resize(simd_lane_);
  for (size_t i = 0; i < ct.size(); ++i) {
    // 1. Ct-Pt Mul and keep result in
    evaluator.multiply_plain_inplace(ct[i], pt[i]);
    InvNttInplace(ct[i], context);
    // 2. Full range masking
    UniformPoly(context, &rnd, ct[i].parms_id());
    // 3. Additive share
    SubPlainInplace(ct[i], rnd, context);
    const auto *rns_tool =
        context.get_context_data(ct[i].parms_id())->rns_tool();
    if (ct[i].parms_id() != final_level_id) {
      evaluator.mod_switch_to_inplace(ct[i], final_level_id);
    }
    // 5. Re-randomize via adding enc(0)
    seal::util::encrypt_zero_asymmetric(public_key, context, ct[i].parms_id(),
                                        ct[i].is_ntt_form(), zero_enc);
    evaluator.add_inplace(ct[i], zero_enc);

    size_t slice_bgn = i * simd_lane_;
    size_t slice_n =
        std::min((size_t)simd_lane_, share_mask.size() - slice_bgn);
    // 6. scale down from Rq => Rt
    rns_tool->decrypt_scale_and_round({rnd.data(), (size_t)simd_lane_},
                                      tmp_poly.data(),
                                      seal::MemoryManager::GetPool());
    // 7. decode Rt as the output share
    DecodeSingle(tmp_poly, share_mask.subspan(slice_bgn, slice_n));

    // 8. Truncate for smaller communication
    if (ct[i].coeff_modulus_size() == 1) {
      TruncateBFVForDecryption(ct[i], context);
    }
  }
}

// Compute ct * pt - mask mod p
void SIMDMulProt::MulThenReshareInplace(absl::Span<RLWECt> ct,
                                        absl::Span<const RLWEPt> pt,
                                        absl::Span<const uint64_t> share_mask,
                                        const RLWEPublicKey &public_key,
                                        const seal::SEALContext &context) {
  SPU_ENFORCE_EQ(ct.size(), pt.size());
  SPU_ENFORCE_EQ(CeilDiv(share_mask.size(), (size_t)simd_lane_), pt.size());

  seal::Evaluator evaluator(context);
  RLWECt zero_enc;
  RLWEPt rnd;

  constexpr int kMarginBitsForDec = 10;
  seal::parms_id_type final_level_id = context.last_parms_id();
  while (final_level_id != context.first_parms_id()) {
    auto cntxt = context.get_context_data(final_level_id);
    if (cntxt->total_coeff_modulus_bit_count() >=
        kMarginBitsForDec + cntxt->parms().plain_modulus().bit_count()) {
      break;
    }
    final_level_id = cntxt->prev_context_data()->parms_id();
  }

  for (size_t i = 0; i < ct.size(); ++i) {
    // 1. Ct-Pt Mul
    evaluator.multiply_plain_inplace(ct[i], pt[i]);

    // 2. Noise flooding
    NoiseFloodInplace(ct[i], context);

    // 3. Drop some modulus for a smaller communication
    evaluator.mod_switch_to_inplace(ct[i], final_level_id);

    // 4. Re-randomize via adding enc(0)
    seal::util::encrypt_zero_asymmetric(public_key, context, ct[i].parms_id(),
                                        ct[i].is_ntt_form(), zero_enc);
    evaluator.add_inplace(ct[i], zero_enc);

    // 5. Additive share
    size_t slice_bgn = i * simd_lane_;
    size_t slice_n =
        std::min((size_t)simd_lane_, share_mask.size() - slice_bgn);
    EncodeSingle(share_mask.subspan(slice_bgn, slice_n), rnd);
    evaluator.sub_plain_inplace(ct[i], rnd);

    // 6. Truncate for smaller communication
    if (ct[i].coeff_modulus_size() == 1) {
      TruncateBFVForDecryption(ct[i], context);
    }
  }
}

// Compute ct0 * pt1 + ct1 * pt1 - mask mod p
void SIMDMulProt::FMAThenReshareInplace(absl::Span<RLWECt> ct0,
                                        absl::Span<const RLWECt> ct1,
                                        absl::Span<const RLWEPt> pt0,
                                        absl::Span<const RLWEPt> pt1,
                                        absl::Span<const uint64_t> share_mask,
                                        const RLWEPublicKey &public_key,
                                        const seal::SEALContext &context) {
  SPU_ENFORCE_EQ(ct0.size(), ct1.size());
  SPU_ENFORCE_EQ(pt0.size(), pt1.size());
  SPU_ENFORCE_EQ(ct0.size(), pt0.size());
  SPU_ENFORCE_EQ(CeilDiv(share_mask.size(), (size_t)simd_lane_), ct0.size());

  seal::Evaluator evaluator(context);
  RLWECt zero_enc;
  RLWEPt rnd;

  constexpr int kMarginBitsForDec = 10;
  seal::parms_id_type final_level_id = context.last_parms_id();
  while (final_level_id != context.first_parms_id()) {
    auto cntxt = context.get_context_data(final_level_id);
    if (cntxt->total_coeff_modulus_bit_count() >=
        kMarginBitsForDec + cntxt->parms().plain_modulus().bit_count()) {
      break;
    }
    final_level_id = cntxt->prev_context_data()->parms_id();
  }

  RLWECt tmp_ct;
  for (size_t i = 0; i < ct0.size(); ++i) {
    // 1. Ct-Pt Mul
    evaluator.multiply_plain_inplace(ct0[i], pt0[i]);
    evaluator.multiply_plain(ct1[i], pt1[i], tmp_ct);
    evaluator.add_inplace(ct0[i], tmp_ct);

    // 2. Noise flooding
    NoiseFloodInplace(ct0[i], context);

    // 3. Drop some modulus for a smaller communication
    evaluator.mod_switch_to_inplace(ct0[i], final_level_id);

    // 4. Re-randomize via adding enc(0)
    seal::util::encrypt_zero_asymmetric(public_key, context, ct0[i].parms_id(),
                                        ct0[i].is_ntt_form(), zero_enc);
    evaluator.add_inplace(ct0[i], zero_enc);

    // 5. Additive share
    size_t slice_bgn = i * simd_lane_;
    size_t slice_n =
        std::min((size_t)simd_lane_, share_mask.size() - slice_bgn);
    EncodeSingle(share_mask.subspan(slice_bgn, slice_n), rnd);
    evaluator.sub_plain_inplace(ct0[i], rnd);

    // 6. Truncate for smaller communication
    if (ct0[i].coeff_modulus_size() == 1) {
      TruncateBFVForDecryption(ct0[i], context);
    }
  }
}

void SIMDMulProt::NoiseFloodInplace(RLWECt &ct,
                                    const seal::SEALContext &context) {
  SPU_ENFORCE(seal::is_metadata_valid_for(ct, context));
  SPU_ENFORCE(ct.size() == 2);
  auto context_data = context.get_context_data(ct.parms_id());
  yacl::CheckNotNull(context_data.get());

  size_t num_coeffs = ct.poly_modulus_degree();
  size_t num_modulus = ct.coeff_modulus_size();

  // e * m for (semi-honest noise) e ~ Gassuain(0, stddev=3.19) and plaintext m
  // \in [-p/2, p/2).
  // |e * m| is bounded by 2*sqrt{N} * 6*stddev * p/2
  size_t noise_bits =
      kNoiseFloodBits + (modulus().bit_count() - 1) +
      std::ceil(
          std::log2(2. * std::sqrt(ct.poly_modulus_degree()) * 6. * 3.19));

  std::vector<uint64_t> wide_noise(num_coeffs * num_modulus);

  // sample r from [-2^{k-1}, 2^{k-1}]
  SampleRanomRNS(absl::MakeSpan(wide_noise), *context_data, noise_bits - 1,
                 ct.is_ntt_form());
  const auto &modulus = context_data->parms().coeff_modulus();

  seal::util::add_poly_coeffmod({wide_noise.data(), num_coeffs},
                                {ct.data(0), num_coeffs}, num_modulus, modulus,
                                {ct.data(0), num_coeffs});
}

}  // namespace spu::mpc::cheetah
