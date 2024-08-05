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

#include "libspu/psi/core/labeled_psi/package.h"

#include <utility>

#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"

namespace spu::psi {

PlainResultPackage ResultPackage::extract(
    const apsi::CryptoContext &crypto_context) {
  SPU_ENFORCE(crypto_context.decryptor(),
              "decryptor is not configured in CryptoContext");

  // SPDLOG_INFO("extract ciphertext");

  seal::Ciphertext psi_result_ct =
      psi_result.extract(crypto_context.seal_context());
  seal::Plaintext psi_result_pt;
  crypto_context.decryptor()->decrypt(psi_result_ct, psi_result_pt);

  SPDLOG_DEBUG(
      "Matching result noise budget: {}",
      crypto_context.decryptor()->invariant_noise_budget(psi_result_ct));

  PlainResultPackage plain_rp;
  plain_rp.bundle_idx = bundle_idx;
  crypto_context.encoder()->decode(psi_result_pt, plain_rp.psi_result);

  plain_rp.label_byte_count = label_byte_count;
  plain_rp.nonce_byte_count = nonce_byte_count;
  for (auto &ct : label_result) {
    seal::Ciphertext label_result_ct =
        ct.extract(crypto_context.seal_context());
    seal::Plaintext label_result_pt;
    crypto_context.decryptor()->decrypt(label_result_ct, label_result_pt);

    std::vector<uint64_t> label_result_data;
    crypto_context.encoder()->decode(label_result_pt, label_result_data);
    plain_rp.label_result.push_back(std::move(label_result_data));
  }

  // Clear the label data
  label_result.clear();

  return plain_rp;
}

}  // namespace spu::psi
