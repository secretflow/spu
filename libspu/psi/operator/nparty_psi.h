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

#include <string>
#include <vector>

#include "yacl/link/link.h"

#include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/operator/base_operator.h"

namespace spu::psi {
// use 2-party psi to get n-party PSI
//   put master rank at 0 position
//   ascending sort other rank by items size,
//   execute ceil( log(n) ) round, get final psi result
// for example 5 party, execute 3 round
//           ||  0    1    2    3   4  5
// 1st round ||  <--------------------->
//           ||       <------------->
//           ||            <---->
//             ======================
//           ||  0    1    2
// 2nd round ||  <--------->
//              =======
//           ||  0    1
// 3rd round ||  <---->
class NpartyPsiOperator : public PsiBaseOperator {
 public:
  enum class PsiProtocol {
    Ecdh,
    Kkrt,
  };
  struct Options {
    std::shared_ptr<yacl::link::Context> link_ctx;

    PsiProtocol psi_proto;
    CurveType curve_type = CurveType::CURVE_25519;
    size_t master_rank = 0;

    // for ecdh
    size_t batch_size = kEcdhPsiBatchSize;
  };

  static Options ParseConfig(const MemoryPsiConfig& config,
                             const std::shared_ptr<yacl::link::Context>& lctx);

 public:
  explicit NpartyPsiOperator(const Options& options);

  std::vector<std::string> OnRun(
      const std::vector<std::string>& inputs) override final;

 private:
  std::vector<std::string> Run2PartyPsi(const std::vector<std::string>& items,
                                        size_t peer_rank, size_t target_rank);

  void GetPsiRank(
      const std::vector<std::pair<size_t, size_t>>& party_size_rank_vec,
      size_t* peer_rank, size_t* target_rank);

  // return item_size-rank pair vector, first element is master, others
  // ascending sort by items size,
  std::vector<std::pair<size_t, size_t>> GetAllPartyItemSizeVec(
      size_t item_size);

 private:
  Options options_;
};

}  // namespace spu::psi