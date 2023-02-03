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

#include "libspu/psi/operator/dp_2party_psi.h"

#include <future>
#include <memory>
#include <random>

#include "spdlog/spdlog.h"
#include "yacl/link/context.h"
#include "yacl/link/link.h"
#include "yacl/utils/serialize.h"

#include "libspu/psi/cryptor/cryptor_selector.h"
#include "libspu/psi/operator/factory.h"

namespace spu::psi {

DpPsiOperator::DpPsiOperator(const std::shared_ptr<yacl::link::Context>& lctx,
                             const DpPsiOptions& options, size_t receiver_rank,
                             CurveType curve_type)
    : PsiBaseOperator(lctx),
      dp_options_(options),
      receiver_rank_(receiver_rank),
      curve_type_(curve_type) {}

std::vector<std::string> DpPsiOperator::OnRun(
    const std::vector<std::string>& inputs) {
  std::vector<std::string> res;

  size_t alice_sub_sample_size = 0;
  size_t alice_up_sample_size = 0;
  size_t bob_sub_sample_size = 0;

  if (receiver_rank_ == link_ctx_->Rank()) {
    std::vector<size_t> dp_psi_result = RunDpEcdhPsiBob(
        dp_options_, link_ctx_, inputs, &bob_sub_sample_size, curve_type_);

    for (auto index : dp_psi_result) {
      res.emplace_back(inputs[index]);
    }
  } else {
    RunDpEcdhPsiAlice(dp_options_, link_ctx_, inputs, &alice_sub_sample_size,
                      &alice_up_sample_size, curve_type_);
  }

  return res;
}

namespace {

std::unique_ptr<PsiBaseOperator> CreateOperator(
    const MemoryPsiConfig& config,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  double bob_sub_sampling = 0.9;
  double epsilon = 3;
  if (config.has_dppsi_params()) {
    bob_sub_sampling = config.dppsi_params().bob_sub_sampling();
    epsilon = config.dppsi_params().epsilon();
  }

  DpPsiOptions dp_options(bob_sub_sampling, epsilon);

  if (config.curve_type() != CurveType::CURVE_INVALID_TYPE) {
    return std::make_unique<DpPsiOperator>(
        lctx, dp_options, config.receiver_rank(), config.curve_type());
  } else {
    return std::make_unique<DpPsiOperator>(lctx, dp_options,
                                           config.receiver_rank());
  }
}

REGISTER_OPERATOR(DP_PSI_2PC, CreateOperator);

}  // namespace

}  // namespace spu::psi
