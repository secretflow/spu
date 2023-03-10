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

#pragma once

#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "yacl/base/byte_container_view.h"
#include "yacl/link/link.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/cipher_store.h"
#include "libspu/psi/utils/ub_psi_cache.h"

// basic ecdh-oprf based psi
// reference:
//  Faster Unbalanced Private Set Intersection
//   https://eprint.iacr.org/2017/677 Fig.1
//  CGT12 Fast and Private Computation of Cardinality of Set Intersection and
//  Union
//   https://eprint.org/2011/141
//
// Unbalanced psi compuation and communication compare reference
// Labeled PSI from Homomorphic Encryption with Reduced Computation and
// Communication Table 2 (https://eprint.iacr.org/2021/1116.pdf)
//
//               server                         client
//                         Offline
//  data shuffle
//  FullEvaluate
//                 send full evaluated items
//                 ----------------------->
// ======================================================
//                         Online
//                                                 Blind
//                    blinded items(batch_size)
//                 <----------------------
//  Evaluate
//                   evaluated items
//                 ----------------------->
//                                              Finalize
// =======================================================
//                                           Intersection
//
namespace spu::psi {

// send queque capacity
inline constexpr size_t kQueueCapacity = 32;
inline constexpr size_t kEcdhOprfPsiBatchSize = 8192;

struct EcdhOprfPsiOptions {
  // Provides the link for server's evaluated data.
  std::shared_ptr<yacl::link::Context> link0;

  // Provides the link for client's blind/evaluated data.
  std::shared_ptr<yacl::link::Context> link1;

  // Now only support 2HashBased Ecdh-OPRF
  OprfType oprf_type = OprfType::Basic;

  // curve_type
  //    FourQ/SM2/Secp256k1
  CurveType curve_type = CurveType::CURVE_FOURQ;

  // batch_size
  //     batch read from IBatchProvider
  //     batch compute oprf blind/evaluate
  //     batch send and read
  size_t batch_size = kEcdhOprfPsiBatchSize;

  // windows_size
  //  control send speed, avoid send buffer overflow
  size_t window_size = kQueueCapacity;
};

class EcdhOprfPsiServer {
 public:
  explicit EcdhOprfPsiServer(const EcdhOprfPsiOptions& options)
      : options_(options),
        oprf_server_(
            CreateEcdhOprfServer(options.oprf_type, options.curve_type)) {}

  EcdhOprfPsiServer(const EcdhOprfPsiOptions& options,
                    yacl::ByteContainerView private_key)
      : options_(options),
        oprf_server_(CreateEcdhOprfServer(private_key, options.oprf_type,
                                          options.curve_type)) {}

  /**
   * @brief FullEvaluate for server side data
   *
   * @param batch_provider input data batch provider
   * @param cipher_store   masked data store
   * @param send_flag  default false, just save to cace,
   *                           true, send and save cache
   */
  size_t FullEvaluate(
      const std::shared_ptr<IShuffleBatchProvider>& batch_provider,
      const std::shared_ptr<IUbPsiCache>& ub_cache, bool send_flag = false);

  /**
   * @brief send masked data
   *
   * @param batch_provider masked data batch provider
   */
  size_t SendFinalEvaluatedItems(
      const std::shared_ptr<IBatchProvider>& batch_provider);

  size_t FullEvaluateAndSend(
      const std::shared_ptr<IShuffleBatchProvider>& batch_provider,
      const std::shared_ptr<IUbPsiCache>& ub_cache = nullptr);

  /**
   * @brief batch recv client blinded items and send evaluate
   *
   */
  void RecvBlindAndSendEvaluate();

  /**
   * @brief batch recv client blinded items and send shuffled evaluate
   *
   */
  void RecvBlindAndShuffleSendEvaluate();

  /**
   * @brief Get the Private Key object
   *
   * @return std::array<uint8_t, kEccKeySize>
   */
  std::array<uint8_t, kEccKeySize> GetPrivateKey() {
    return oprf_server_->GetPrivateKey();
  }

  size_t GetCompareLength() { return oprf_server_->GetCompareLength(); }

  std::pair<std::vector<uint64_t>, size_t> RecvIntersectionMaskedItems(
      const std::shared_ptr<IShuffleBatchProvider>& cache_provider,
      size_t batch_size);

 private:
  EcdhOprfPsiOptions options_;

  std::shared_ptr<IEcdhOprfServer> oprf_server_;
};

class EcdhOprfPsiClient {
 public:
  explicit EcdhOprfPsiClient(const EcdhOprfPsiOptions& options)
      : options_(options) {
    std::shared_ptr<IEcdhOprfClient> oprf_client =
        CreateEcdhOprfClient(options.oprf_type, options.curve_type);
    compare_length_ = oprf_client->GetCompareLength();
    ec_point_length_ = oprf_client->GetEcPointLength();
  }

  explicit EcdhOprfPsiClient(const EcdhOprfPsiOptions& options,
                             yacl::ByteContainerView private_key)
      : options_(options) {
    oprf_client_ = CreateEcdhOprfClient(private_key, options.oprf_type,
                                        options.curve_type);
    compare_length_ = oprf_client_->GetCompareLength();
    ec_point_length_ = oprf_client_->GetEcPointLength();
  }

  /**
   * @brief recv server's masked data
   *
   * @param cipher_store store server's masked data to peer results
   */
  void RecvFinalEvaluatedItems(
      const std::shared_ptr<ICipherStore>& cipher_store);

  /**
   * @brief blind input data and send to server
   *
   * @param batch_provider input data batch provider
   */
  size_t SendBlindedItems(
      const std::shared_ptr<IBatchProvider>& batch_provider);

  /**
   * @brief recv evaluated data, do Finalize and store to cipher_store
   *
   * @param batch_provider  input data batch provider
   * @param cipher_store    store finalized data to self results
   */
  void RecvEvaluatedItems(const std::shared_ptr<ICipherStore>& cipher_store);

  void SendIntersectionMaskedItems(
      const std::shared_ptr<IBatchProvider>& batch_provider);

 private:
  EcdhOprfPsiOptions options_;

  std::mutex mutex_;
  std::condition_variable queue_push_cv_;
  std::condition_variable queue_pop_cv_;
  std::queue<std::vector<std::shared_ptr<IEcdhOprfClient>>> oprf_client_queue_;
  std::shared_ptr<IEcdhOprfClient> oprf_client_ = nullptr;

  size_t compare_length_;
  size_t ec_point_length_;
};

}  // namespace spu::psi
