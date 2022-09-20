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

#include "spu/psi/operator/bc22_2party_psi.h"

#include "spu/psi/core/bc22_psi/bc22_psi.h"

namespace spu::psi {

Bc22PcgPsiOperator::Bc22PcgPsiOperator(const Options& options)
    : PsiBaseOperator(options.lctx), options_(options) {}

std::vector<std::string> Bc22PcgPsiOperator::OnRun(
    const std::vector<std::string>& inputs) {
  auto role = link_ctx_->Rank() == options_.receiver_rank
                  ? PsiRoleType::Receiver
                  : PsiRoleType::Sender;
  Bc22PcgPsi pcg_psi(link_ctx_, role);
  pcg_psi.RunPsi(inputs);
  if (role == PsiRoleType::Receiver) {
    return pcg_psi.GetIntersection();
  } else {
    return {};
  }
}

}  // namespace spu::psi