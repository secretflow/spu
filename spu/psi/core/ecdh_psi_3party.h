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

#include "spu/psi/core/ecdh_psi_mparty.h"

namespace spu::psi {

//
// 3party ecdh psi algorithm.
//
//  master get the final intersection
//     alice role is master-link->NextRank()
//     bob   role is master-link->PrevRank()
//
//     alice           bob                    candy(master)
// step 1 =============================================
//       |              | shuffle b items       |
//  x^a  |      x^a     |                       |
//       |   -------->  | x^a^b                 |
//       |              |                       |
//       |      y^b     | y^b                   |
//       |   <--------  |                       |
// y^b^a |              |                       |
// step 2 ==============================================
//       |              | shuffle x^a^b         |
//       |      x^a^b   |                       |
//       |   <--------  |                       |
// calc intersection_ab |                       |
//       |            intersection_ab           |
//       |   ---------------------------------> | {intersection_ab}^c
// step 3 ==============================================
//       |                 z^c                  |
//       |   <--------------------------------- |
// z^c^a |              |                       |
//       |    z^c^a     |                       |
//       |  -------->   |                       |
//       |              | calc {z^c^a}^b        |
//       |              |  send z^c^a^b         |
//       |              |  ------------------>  |
//       |              |                       |
//                                      calc intersection_abc
class ShuffleEcdhPSI3Party {
 public:
  /**
   * @brief Construct a new ShuffleEcdhPSI3Party object
   *
   * @param link    network communication link context
   * @param master_rank  rank of master role the 3party psi protocol
   * @param batch_provider    data provider which produce batch of strings.
   * @param cipher_store  CipherStore stores dual encrypted results.
   * @param batch_size   mask/send size every batch, default kEcdhPsiBatchSize
   */
  ShuffleEcdhPSI3Party(const std::shared_ptr<yasl::link::Context>& link,
                       size_t master_rank,
                       const std::shared_ptr<IBatchProvider>& batch_provider,
                       const std::shared_ptr<ICipherStore>& cipher_store,
                       CurveType curve_type = CurveType::Curve25519,
                       size_t batch_size = kEcdhPsiBatchSize);

  /**
   * @brief Construct a new ShuffleEcdhPSI3Party object
   *
   * @param link    network communication link context
   * @param master_rank  rank of master role the 3party psi protocol
   * @param items    input data for psi
   * @param cipher_store  ICipherStore stores dual encrypted results.
   * @param batch_size   mask/send size every batch, default kEcdhPsiBatchSize
   */
  ShuffleEcdhPSI3Party(const std::shared_ptr<yasl::link::Context>& link,
                       size_t master_rank,
                       const std::vector<std::string>& items,
                       const std::shared_ptr<ICipherStore>& cipher_store,
                       CurveType curve_type = CurveType::Curve25519,
                       size_t batch_size = kEcdhPsiBatchSize);

  /**
   * @brief  first step of three step show in the flow
   *
   * @return null
   */

  void RunEcdhPsiStep1();

  /**
   * @brief second step of three step show in the flow
   *
   * @param batch_provider  data provider after shuffle action
   */
  void RunEcdhPsiStep2(
      const std::shared_ptr<IBatchProvider>& batch_provider = nullptr);

  /**
   * @brief third step of three step show in the flow
   *
   * @param batch_provider  data provider after two party intersection action
   */
  void RunEcdhPsiStep3(
      const std::shared_ptr<IBatchProvider>& batch_provider = nullptr);

 private:
  void RunEcdhPsiMaster();
  void RunEcdhPsiSlave();

 private:
  std::shared_ptr<yasl::link::Context> link_;
  EcdhPsiMParty ecdh_psi_;
  size_t master_rank_;
};

//
//
/**
 * @brief Simple wrapper for a common in memory shuffle based-psi case.
 *
 * @param link    network communication link context
 * @param master_rank  rank of master role the 3party psi protocol
 * @param items    input data for psi
 * @param batch_size   mask/send size every batch, default kEcdhPsiBatchSize
 * @return std::vector<std::string>  master_rank get intersection result, other
 * rank return empty vector
 */
std::vector<std::string> RunShuffleEcdhPsi3Party(
    const std::shared_ptr<yasl::link::Context>& link, size_t master_rank,
    std::vector<std::string>& items,
    CurveType curve_type = CurveType::Curve25519,
    size_t batch_size = kEcdhPsiBatchSize);

}  // namespace spu::psi
