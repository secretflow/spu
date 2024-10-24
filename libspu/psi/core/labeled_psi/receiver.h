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

#pragma once

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "apsi/crypto_context.h"
#include "apsi/itt.h"
#include "apsi/powers.h"
#include "apsi/psi_params.h"
#include "apsi/seal_object.h"
#include "gsl/span"
#include "seal/seal.h"
#include "yacl/link/link.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf.h"

#include "libspu/psi/core/labeled_psi/serializable.pb.h"

namespace spu::psi {

class LabelPsiReceiver {
 public:
  explicit LabelPsiReceiver(const apsi::PSIParams &params,
                            bool has_label = false);

  /**
   * @brief  Request PSI Parameters
   *
   * @param items_size  receiver's items size
   * @param link_ctx    link context
   * @return apsi::PSIParams
   */
  static apsi::PSIParams RequestPsiParams(
      size_t items_size, const std::shared_ptr<yacl::link::Context> &link_ctx);

  /**
   * @brief  Request items oprf
   *
   * @param items    receiver's items
   * @param link_ctx link context
   * @return std::pair<std::vector<apsi::HashedItem>,
   * std::vector<apsi::LabelKey>>
   *
   * split items's oprf(32B) to HashedItem(16B) and LabelKey(16B)
   */
  static std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>
  RequestOPRF(const std::vector<std::string> &items,
              const std::shared_ptr<yacl::link::Context> &link_ctx);

  /**
   * @brief  Request PSI Query
   *         send items's query_powers
   *
   * @param hashed_items
   * @param label_keys
   * @param link_ctx
   * @return std::pair<std::vector<size_t>, std::vector<std::string>>
   *  Get query polynomial ciphertext
   *
   */
  std::pair<std::vector<size_t>, std::vector<std::string>> RequestQuery(
      const std::vector<apsi::HashedItem> &hashed_items,
      const std::vector<apsi::LabelKey> &label_keys,
      const std::shared_ptr<yacl::link::Context> &link_ctx);

  /**
  Generates a new set of keys to use for queries.
  */
  void ResetKeys();

  /**
  Returns a reference to the PowersDag configured for this Receiver.
  */
  const apsi::PowersDag &GetPowersDag() const { return pd_; }

  /**
  Returns a reference to the CryptoContext for this Receiver.
  */
  const apsi::CryptoContext &GetCryptoContext() const {
    return crypto_context_;
  }

  /**
  Returns a reference to the SEALContext for this Receiver.
  */
  std::shared_ptr<seal::SEALContext> GetSealContext() const {
    return crypto_context_.seal_context();
  }

 private:
  /**
  Recomputes the PowersDag. The function returns the depth of the
  PowersDag. In some cases the receiver may want to ensure that the depth of
  the powers computation will be as expected (PowersDag::depth), and
  otherwise attempt to reconfigure the PowersDag.
  */
  std::uint32_t ResetPowersDag(const std::set<std::uint32_t> &source_powers);

  void Initialize();

  std::vector<std::pair<size_t, std::string>> ProcessQueryResult(
      const proto::QueryResultProto &query_result_proto,
      const apsi::receiver::IndexTranslationTable &itt,
      const std::vector<apsi::LabelKey> &label_keys);

  apsi::PSIParams psi_params_;

  apsi::CryptoContext crypto_context_;

  apsi::PowersDag pd_;

  apsi::SEALObject<seal::RelinKeys> relin_keys_;

  // NOTE(juhou): we now support zstd compression by default
  seal::compr_mode_type compr_mode_ = seal::Serialization::compr_mode_default;

  bool has_label_;
};
}  // namespace spu::psi
