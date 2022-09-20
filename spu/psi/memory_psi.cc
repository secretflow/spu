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

#include "spu/psi/memory_psi.h"

#include "spdlog/spdlog.h"

#include "spu/psi/core/ecdh_psi.h"
#include "spu/psi/operator/operator.h"
#include "spu/psi/utils/utils.h"

namespace spu::psi {

MemoryPsi::MemoryPsi(MemoryPsiConfig config,
                     std::shared_ptr<yasl::link::Context> lctx)
    : config_(std::move(config)), lctx_(std::move(lctx)) {
  CheckOptions();
}

void MemoryPsi::CheckOptions() const {
  // options sanity check.
  YASL_ENFORCE(config_.psi_type() != PsiType::INVALID_PSI_TYPE,
               "unsupported psi proto:{}", config_.psi_type());

  YASL_ENFORCE(
      static_cast<size_t>(config_.receiver_rank()) < lctx_->WorldSize(),
      "invalid receiver_rank:{}, world_size:{}", config_.receiver_rank(),
      lctx_->WorldSize());

  // check world size
  if (config_.psi_type() == PsiType::ECDH_PSI_2PC ||
      config_.psi_type() == PsiType::KKRT_PSI_2PC ||
      config_.psi_type() == PsiType::BC22_PSI_2PC) {
    YASL_ENFORCE(lctx_->WorldSize() == 2,
                 "psi_type:{}, only two parties supported, got "
                 "{}",
                 config_.psi_type(), lctx_->WorldSize());
  }
  if (config_.psi_type() == PsiType::ECDH_PSI_3PC) {
    if (lctx_->WorldSize() != 3) {
      YASL_ENFORCE(lctx_->WorldSize() == 3,
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

  auto run_psi_f = std::async([&] {
    // TODO: auto register operator
    if ((config_.psi_type() == PsiType::KKRT_PSI_NPC) ||
        (config_.psi_type() == PsiType::ECDH_PSI_NPC)) {
      res = NPartyPsi(inputs);
    } else if (config_.psi_type() == PsiType::ECDH_PSI_3PC) {
      res = Ecdh3PartyPsi(inputs);
    } else if (config_.psi_type() == PsiType::KKRT_PSI_2PC) {
      res = KkrtPsi(inputs);
    } else if (config_.psi_type() == PsiType::ECDH_PSI_2PC) {
      res = EcdhPsi(inputs);
    } else if (config_.psi_type() == PsiType::BC22_PSI_2PC) {
      res = Bc22Psi(inputs);
    } else {
      YASL_THROW("logistic error, psi_type={}", (config_.psi_type()));
    }
  });
  SyncWait(lctx_, &run_psi_f);

  return res;
}

std::vector<std::string> MemoryPsi::EcdhPsi(
    const std::vector<std::string>& inputs) {
  size_t target_rank = config_.receiver_rank();
  if (config_.broadcast_result()) {
    target_rank = yasl::link::kAllRank;
  }

  if (config_.curve_type() != CurveType::CURVE_INVALID_TYPE) {
    return RunEcdhPsi(lctx_, inputs, target_rank, config_.curve_type());
  }
  return RunEcdhPsi(lctx_, inputs, target_rank);
}

std::vector<std::string> MemoryPsi::KkrtPsi(
    const std::vector<std::string>& inputs) {
  KkrtPsiOperator::Options opts;
  opts.link_ctx = lctx_;
  opts.receiver_rank = config_.receiver_rank();

  SPDLOG_INFO("kkrt psi receiver_rank={}, rank={}", opts.receiver_rank,
              lctx_->Rank());

  auto op = CreatePsiOperator(opts);
  return op->Run(inputs, config_.broadcast_result());
}

std::vector<std::string> MemoryPsi::NPartyPsi(
    const std::vector<std::string>& inputs) {
  NpartyPsiOperator::Options opts;
  opts.link_ctx = lctx_;
  opts.master_rank = config_.receiver_rank();
  opts.psi_type = NpartyPsiOperator::PsiType::Ecdh;
  if (config_.psi_type() == PsiType::KKRT_PSI_NPC) {
    opts.psi_type = NpartyPsiOperator::PsiType::Kkrt;
  }
  if (config_.curve_type() != CurveType::CURVE_INVALID_TYPE) {
    opts.curve_type = config_.curve_type();
  }

  auto op = CreatePsiOperator(opts);
  return op->Run(inputs, config_.broadcast_result());
}

std::vector<std::string> MemoryPsi::Ecdh3PartyPsi(
    const std::vector<std::string>& inputs) {
  Ecdh3PartyPsiOperator::Options opts;
  opts.link_ctx = lctx_;
  opts.master_rank = config_.receiver_rank();
  if (config_.curve_type() != CurveType::CURVE_INVALID_TYPE) {
    opts.curve_type = config_.curve_type();
  }

  auto op = CreatePsiOperator(opts);
  return op->Run(inputs, config_.broadcast_result());
}

std::vector<std::string> MemoryPsi::Bc22Psi(
    const std::vector<std::string>& inputs) {
  Bc22PcgPsiOperator::Options opts;
  opts.lctx = lctx_;
  opts.receiver_rank = config_.receiver_rank();
  auto op = CreatePsiOperator(opts);
  return op->Run(inputs, config_.broadcast_result());
}

}  // namespace spu::psi
