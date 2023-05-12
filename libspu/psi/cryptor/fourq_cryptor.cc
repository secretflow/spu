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

#include "libspu/psi/cryptor/fourq_cryptor.h"

#include <vector>

#include "apsi/fourq/FourQ_api.h"
#include "apsi/fourq/FourQ_internal.h"
#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/utils/parallel.h"

namespace spu::psi {

void FourQEccCryptor::EccMask(absl::Span<const char> batch_points,
                              absl::Span<char> dest_points) const {
  SPU_ENFORCE(batch_points.size() % kEccKeySize == 0);

  using Item = std::array<unsigned char, kEccKeySize>;
  static_assert(sizeof(Item) == kEccKeySize);

  auto mask_functor = [this](const Item& in, Item& out) {
    ECCRYPTO_STATUS status =
        CompressedSecretAgreement(this->private_key_, in.data(), out.data());

    SPU_ENFORCE(status == ECCRYPTO_SUCCESS,
                "FourQ CompressedSecretAgreement Error: ", status);
  };

  absl::Span<const Item> input(
      reinterpret_cast<const Item*>(batch_points.data()),
      batch_points.size() / sizeof(Item));
  absl::Span<Item> output(reinterpret_cast<Item*>(dest_points.data()),
                          dest_points.size() / sizeof(Item));

  yacl::parallel_for(0, input.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      mask_functor(input[idx], output[idx]);
    }
  });
}

std::vector<uint8_t> FourQEccCryptor::HashToCurve(
    absl::Span<const char> input) const {
  point_t P;
  std::vector<uint8_t> sha_bytes =
      yacl::crypto::SslHash(yacl::crypto::HashAlgorithm::SHA512)
          .Update(input)
          .CumulativeHash();
  auto* f2elmt = reinterpret_cast<f2elm_t*>(sha_bytes.data());
  mod1271((reinterpret_cast<felm_t*>(f2elmt))[0]);
  mod1271((reinterpret_cast<felm_t*>(f2elmt))[1]);
  ECCRYPTO_STATUS status = ECCRYPTO_SUCCESS;
  // Hash GF(p^2) element to curve
  status = ::HashToCurve(reinterpret_cast<felm_t*>(f2elmt), P);
  SPU_ENFORCE(status == ECCRYPTO_SUCCESS, "FourQ HashToCurve Error: ", status);
  std::vector<uint8_t> ret(kEccKeySize, 0);
  encode(P, static_cast<unsigned char*>(ret.data()));  // Encode public key
  return ret;
}

}  // namespace spu::psi
