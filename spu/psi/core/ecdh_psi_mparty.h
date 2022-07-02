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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "yasl/link/link.h"

#include "spu/psi/core/ecdh_psi.h"

namespace spu::psi {

// mparty ecdh psi.
//
// class for mparty ecdh psi
//  common element:
//    private_key_:  ec(curve25519) private key, 32B
//    options_:       PsiOption
class EcdhPsiMParty {
 public:
  /**
   * @brief Construct a new EcdhPsi object
   *
   * @param batch_provider  data source for batch read
   * @param cipher_store    CipherStore for peer and self mask
   * @param batch_size      batch read item size
   */
  EcdhPsiMParty(const std::shared_ptr<IBatchProvider>& batch_provider,
                const std::shared_ptr<ICipherStore>& cipher_store,
                CurveType curve_type = CurveType::Curve25519,
                size_t batch_size = kEcdhPsiBatchSize);

  /**
   * @brief Construct a new EcdhPsi object
   *
   * @param items  data source vector
   * @param cipher_store    CipherStore for peer and self mask
   */
  EcdhPsiMParty(const std::vector<std::string>& items,
                const std::shared_ptr<ICipherStore>& cipher_store,
                CurveType curve_type = CurveType::Curve25519,
                size_t batch_size = kEcdhPsiBatchSize);

  ~EcdhPsiMParty();

  /**
   * @brief Get the Private Key object
   *
   * @return std::vector<uint8_t>
   */
  std::vector<uint8_t> GetPrivateKey() const { return private_key_; }

  /**
   * @brief mask self items and send to send_rank
   *  (1) calc x^a,
   *  (2) send x^a to link's send_rank
   *
   * @param link         network context
   * @param send_rank    send target rank
   */
  void RunMaskSelfAndSend(const std::shared_ptr<yasl::link::Context>& link,
                          size_t send_rank);

  /**
   * @brief recv from recv_rank, mask and send to send_rank
   *         x^a
   *    ------------>  |
   *                   |  (1) recv x^a from recv_rank
   *                   |  (2) calc x^a^b, truncate dual_mask_size byte
   *                   |  (3) send x^a^b to send_rank
   *                   | ----------------------------->
   *
   * @param link            network context
   * @param recv_rank       recv from link's recv_rank
   * @param send_rank       forward to send_rank
   * @param dual_mask_size  store items data size, default KHashSize
   *   final compare use kFinalCompareBytes
   */
  void RunMaskRecvAndForward(const std::shared_ptr<yasl::link::Context>& link,
                             size_t recv_rank, size_t send_rank,
                             size_t dual_mask_size = kHashSize);

  /**
   * @brief recv from link's recv_rank, mask and store to cipher_batch,
   * after store recieved data, can do shuffle with stored data，
   * length depend on dual_mask_size
   *        x^a
   *    ------------>  |
   *                   |  (1) recv x^a from recv_rank
   *                   |  (2) calc x^a^b, truncate dual_mask_size byte
   *                   |  (3) store x^a^b to cipher_batch's peer results
   *
   * @param link            network context
   * @param recv_rank       recv from link's recv_rank
   * @param dual_mask_size  store items data size, default KHashSize
   *   final compare use kFinalCompareBytes
   */
  void RunMaskRecvAndStore(const std::shared_ptr<yasl::link::Context>& link,
                           size_t recv_rank, size_t dual_mask_size = kHashSize);

  /**
   * @brief send data from provider to send_rank
   * batch_provider's data after intersection or shuffle
   *   (1) batch read data from batch_provider
   *   (2) send to link's send_rank
   *                  --------------------------->
   *
   * @param link            network context
   * @param send_rank       send to link's send_rank
   * @param batch_provider  data provider for batch read
   */
  void RunSendBatch(const std::shared_ptr<yasl::link::Context>& link,
                    size_t send_rank,
                    const std::shared_ptr<IBatchProvider>& batch_provider);

  /**
   * @brief recv from recv_rank, store to cipherStore
   * items data length depend on dual_mask_size
   * ----------->
   *              (1) batch recv from link's recv_rank
   *              (2) store received items to cipher_batch's self results
   *
   * @param link            network context
   * @param recv_rank       recv from link's recv_rank
   * @param dual_mask_size  store items data size, default KHashSize
   *   final compare use kFinalCompareBytes
   */
  void RunRecvAndStore(const std::shared_ptr<yasl::link::Context>& link,
                       size_t recv_rank, size_t dual_mask_size = kHashSize);

 protected:
  // curve 25519 dh private key, 32B
  std::vector<uint8_t> private_key_;

  // psi option
  PsiOptions options_;
};

}  // namespace spu::psi
