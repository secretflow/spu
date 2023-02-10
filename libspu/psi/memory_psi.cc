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

#include "libspu/psi/memory_psi.h"

#include "spdlog/spdlog.h"

#include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/operator/factory.h"
#include "libspu/psi/utils/utils.h"

namespace spu::psi {

MemoryPsi::MemoryPsi(MemoryPsiConfig config,
                     std::shared_ptr<yacl::link::Context> lctx)
    : config_(std::move(config)), lctx_(std::move(lctx)) {
  CheckOptions();
}

void MemoryPsi::CheckOptions() const {
  // options sanity check.
  SPU_ENFORCE(config_.psi_type() != PsiType::INVALID_PSI_TYPE,
              "unsupported psi proto:{}", config_.psi_type());

  SPU_ENFORCE(static_cast<size_t>(config_.receiver_rank()) < lctx_->WorldSize(),
              "invalid receiver_rank:{}, world_size:{}",
              config_.receiver_rank(), lctx_->WorldSize());

  // check world size
  if (config_.psi_type() == PsiType::ECDH_PSI_2PC ||
      config_.psi_type() == PsiType::KKRT_PSI_2PC ||
      config_.psi_type() == PsiType::BC22_PSI_2PC) {
    SPU_ENFORCE(lctx_->WorldSize() == 2,
                "psi_type:{}, only two parties supported, got "
                "{}",
                config_.psi_type(), lctx_->WorldSize());
  }
  if (config_.psi_type() == PsiType::ECDH_PSI_3PC) {
    if (lctx_->WorldSize() != 3) {
      SPU_ENFORCE(lctx_->WorldSize() == 3,
                  "psi_type:{}, only three parties supported, got "
                  "{}",
                  config_.psi_type(), lctx_->WorldSize());
    }
  }
}

std::vector<std::string> MemoryPsi::Run(
    const std::vector<std::string>& inputs) {
  std::vector<std::string> res;
  size_t min_inputs_size = inputs.size();
  std::vector<size_t> inputs_size_list =
      AllGatherItemsSize(lctx_, inputs.size());
  for (size_t idx = 0; idx < inputs_size_list.size(); idx++) {
    SPDLOG_INFO("psi protocol={}, rank={}, inputs_size={}", config_.psi_type(),
                idx, inputs_size_list[idx]);
    min_inputs_size = std::min(min_inputs_size, inputs_size_list[idx]);
  }
  if (min_inputs_size == 0) {
    SPDLOG_INFO(
        "psi protocol={}, min_inputs_size=0, "
        "no need do intersection",
        config_.psi_type());
    return res;
  }

  if (config_.psi_type() == PsiType::ECDH_PSI_2PC) {
    res = EcdhPsi(inputs);
  } else {
    res = OperatorFactory::GetInstance()
              ->Create(config_, lctx_)
              ->Run(inputs, config_.broadcast_result());
  }

  return res;
}

std::vector<std::string> MemoryPsi::EcdhPsi(
    const std::vector<std::string>& inputs) {
  size_t target_rank = config_.receiver_rank();
  if (config_.broadcast_result()) {
    target_rank = yacl::link::kAllRank;
  }

  if (config_.curve_type() != CurveType::CURVE_INVALID_TYPE) {
    return RunEcdhPsi(lctx_, inputs, target_rank, config_.curve_type());
  }
  return RunEcdhPsi(lctx_, inputs, target_rank);
}

}  // namespace spu::psi
