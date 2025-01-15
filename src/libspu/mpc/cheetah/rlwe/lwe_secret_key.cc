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

LWESecretKey::LWESecretKey(const RLWESecretKey &rlwe_sk,
                           const seal::SEALContext &context) {
  SPU_ENFORCE(seal::is_metadata_valid_for(rlwe_sk, context),
              "invalid rlwe secret key for this context");

  const auto &parms = context.key_context_data()->parms();
  const auto &modulus = parms.coeff_modulus();
  const size_t num_coeff = parms.poly_modulus_degree();
  const size_t num_modulus = modulus.size();

  // NOTE(juhou) need parms_id_zero before resize()
  secret_non_ntt_.data().parms_id() = seal::parms_id_zero;
  secret_non_ntt_.data().resize(num_coeff * num_modulus);
  secret_non_ntt_.data().parms_id() = rlwe_sk.data().parms_id();

  std::copy_n(rlwe_sk.data().data(), num_coeff * num_modulus,
              secret_non_ntt_.data().data());

  if (rlwe_sk.data().is_ntt_form()) {
    const auto *ntt_tables = context.key_context_data()->small_ntt_tables();
    auto *sk_ptr = secret_non_ntt_.data().data();
    // we keep the LWESecretKey in the non-ntt form
    for (size_t l = 0; l < num_modulus; ++l, sk_ptr += num_coeff) {
      seal::util::inverse_ntt_negacyclic_harvey(sk_ptr, ntt_tables[l]);
    }
  }
}

LWESecretKey::~LWESecretKey() = default;

size_t LWESecretKey::save_size(seal::compr_mode_type compr_mode) const {
  return secret_non_ntt_.save_size(compr_mode);
}

size_t LWESecretKey::save(seal::seal_byte *buffer, size_t size,
                          seal::compr_mode_type compr_mode) const {
  try {
    return secret_non_ntt_.save(buffer, size, compr_mode);
  } catch (const std::exception &e) {
    SPU_THROW(fmt::format("SEAL error [{}]", e.what()));
  }
}

void LWESecretKey::load(const seal::SEALContext &context,
                        const seal::seal_byte *buffer, size_t size) {
  try {
    secret_non_ntt_.load(context, buffer, size);
  } catch (const std::exception &e) {
    SPU_THROW(fmt::format("SEAL error [{}]", e.what()));
  }
}

void LWESecretKey::unsafe_load(const seal::SEALContext &context,
                               const seal::seal_byte *buffer, size_t size) {
  try {
    secret_non_ntt_.unsafe_load(context, buffer, size);
  } catch (const std::exception &e) {
    SPU_THROW(fmt::format("SEAL error [{}]", e.what()));
  }
}

}  // namespace spu::mpc::cheetah
