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
#include <string>
#include <vector>

#include "libspu/psi/core/ecdh_3pc_psi.h"
#include "libspu/psi/operator/base_operator.h"

namespace spu::psi {

//
// 3party ecdh psi algorithm.
//
//  master get the final intersection
//     calcuator role is master-link->NextRank()
//     partner   role is master-link->PrevRank()
//
//  (calcuator)       (partner)             (master)
//     alice           bob                    candy
// partners psi =============================================
//       |              | shuffle b items       |
//  x^a  |      x^a     |                       |
//       |   -------->  | x^a^b                 |
//       |              |                       |
//       |      y^b     | y^b                   |
//       |   <--------  |                       |
// y^b^a |              |                       |
//       |              | shuffle x^a^b         |
//       |      x^a^b   |                       |
//       |   <--------  |                       |
// calc intersection_ab |                       |
//       |            intersection_ab           |
//       |   ---------------------------------> | {intersection_ab}^c
// mask master ==============================================
//       |                 z^c                  |
//       |   <--------------------------------- |
// z^c^a |              |                       |
//       |    z^c^a     |                       |
//       |  -------->   |                       |
//       |              | calc {z^c^a}^b        |
//       |              |  send z^c^a^b         |
//       |              |  ------------------>  |
// calc result ==============================================
//       |              |                       |
//                                      calc intersection_abc
class Ecdh3PartyPsiOperator : public PsiBaseOperator {
 public:
  struct Options {
    // Provides the link for the rank world.
    std::shared_ptr<yacl::link::Context> link_ctx;

    size_t master_rank;

    // batch_size
    //     batch compute dh mask
    //     batch send and read
    size_t batch_size = kEcdhPsiBatchSize;

    size_t dual_mask_size = kFinalCompareBytes;

    // curve_type
    CurveType curve_type = CurveType::CURVE_25519;
  };

  static Options ParseConfig(const MemoryPsiConfig& config,
                             const std::shared_ptr<yacl::link::Context>& lctx);

 public:
  explicit Ecdh3PartyPsiOperator(const Options& options);

 public:
  std::vector<std::string> OnRun(
      const std::vector<std::string>& inputs) override final;

 private:
  Options options_;

  std::shared_ptr<ShuffleEcdh3PcPsi> handler_;
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
 * @return std::vector<std::string>  master_rank get intersection result,
 * other rank return empty vector
 */
std::vector<std::string> RunShuffleEcdh3PartyPsi(
    const std::shared_ptr<yacl::link::Context>& link, size_t master_rank,
    std::vector<std::string>& items,
    CurveType curve_type = CurveType::CURVE_25519,
    size_t batch_size = kEcdhPsiBatchSize);

}  // namespace spu::psi
