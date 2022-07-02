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
#include <queue>
#include <string>
#include <vector>

#include "yasl/base/byte_container_view.h"
#include "yasl/link/link.h"

#include "spu/psi/core/ecdh_psi.h"
#include "spu/psi/cryptor/ecdh_oprf/ecdh_oprf.h"
#include "spu/psi/cryptor/ecdh_oprf/ecdh_oprf_selector.h"

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
namespace spu {
namespace psi {

// send queque capacity
inline constexpr size_t kQueueCapacity = 32;

struct EcdhOprfPsiOptions {
  // Provides the link for server's evaluated data.
  std::shared_ptr<yasl::link::Context> link0;

  // Provides the link for client's blind/evaluated data.
  std::shared_ptr<yasl::link::Context> link1;

  // Now only support 2HashBased Ecdh-OPRF
  OprfType oprf_type = OprfType::Basic;

  // curve_type
  //    FourQ/SM2/Secp256k1
  CurveType curve_type = CurveType::CurveFourQ;

  // batch_size
  //     batch read from IBatchProvider
  //     batch compute oprf blind/evaluate
  //     batch send and read
  size_t batch_size = kEcdhPsiBatchSize;

  // windows_size
  //  control send speed, avoid send buffer overflow
  size_t window_size = kQueueCapacity;
};

class EcdhOprfPsiServer {
 public:
  EcdhOprfPsiServer(EcdhOprfPsiOptions options)
      : options_(options),
        oprf_server_(
            CreateEcdhOprfServer(options.oprf_type, options.curve_type)) {}

  EcdhOprfPsiServer(EcdhOprfPsiOptions options,
                    yasl::ByteContainerView private_key)
      : options_(options),
        oprf_server_(CreateEcdhOprfServer(private_key, options.oprf_type,
                                          options.curve_type)) {}

  /**
   * @brief FullEvaluate for server side data
   *
   * @param batch_provider input data batch provider
   * @param cipher_store   masked data store
   */
  void FullEvaluate(std::shared_ptr<IBatchProvider> batch_provider,
                    std::shared_ptr<ICipherStore> cipher_store);

  /**
   * @brief send masked data
   *
   * @param batch_provider masked data batch provider
   */
  void SendFinalEvaluatedItems(std::shared_ptr<IBatchProvider> batch_provider);

  /**
   * @brief
   *
   */
  void RecvBlindAndSendEvaluate();

  /**
   * @brief Get the Private Key object
   *
   * @return std::array<uint8_t, kEccKeySize>
   */
  std::array<uint8_t, kEccKeySize> GetPrivateKey() {
    return oprf_server_->GetPrivateKey();
  }

 private:
  EcdhOprfPsiOptions options_;

  std::shared_ptr<IEcdhOprfServer> oprf_server_;
};

class EcdhOprfPsiClient {
 public:
  EcdhOprfPsiClient(EcdhOprfPsiOptions options) : options_(options) {
    std::shared_ptr<IEcdhOprfClient> oprf_client =
        CreateEcdhOprfClient(options.oprf_type, options.curve_type);
    compare_length_ = oprf_client->GetCompareLength();
    ec_point_length_ = oprf_client->GetEcPointLength();
  }

  /**
   * @brief recv server's masked data
   *
   * @param cipher_store store server's masked data to peer results
   */
  void RecvFinalEvaluatedItems(std::shared_ptr<ICipherStore> cipher_store);

  /**
   * @brief blind input data and send to server
   *
   * @param batch_provider input data batch provider
   */
  void SendBlindedItems(std::shared_ptr<IBatchProvider> batch_provider);

  /**
   * @brief recv evaluated data, do Finalize and store to cipher_store
   *
   * @param batch_provider  input data batch provider
   * @param cipher_store    store finalized data to self results
   */
  void RecvEvaluatedItems(std::shared_ptr<IBatchProvider> batch_provider,
                          std::shared_ptr<ICipherStore> cipher_store);

 private:
  EcdhOprfPsiOptions options_;

  std::mutex mutex_;
  std::condition_variable queue_push_cv_;
  std::condition_variable queue_pop_cv_;
  std::queue<std::vector<std::shared_ptr<IEcdhOprfClient>>> oprf_client_queue_;

  size_t compare_length_;
  size_t ec_point_length_;
};
}  // namespace psi
}  // namespace spu