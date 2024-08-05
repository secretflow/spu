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

#include "libspu/core/context.h"

namespace spu {
namespace {

std::string genRootObjectId(const std::shared_ptr<yacl::link::Context>& lctx) {
  // In single-process simulation mode, multi-context need to use different id,
  // or tracing will not work.
  if (lctx) {
    return fmt::format("root-{}", lctx->Rank());
  }

  return "root";
}

}  // namespace

SPUContext::SPUContext(const RuntimeConfig& config,
                       const std::shared_ptr<yacl::link::Context>& lctx)
    : config_(config),
      prot_(std::make_unique<Object>(genRootObjectId(lctx))),
      lctx_(lctx) {}

std::unique_ptr<SPUContext> SPUContext::fork() const {
  std::shared_ptr<yacl::link::Context> new_lctx =
      lctx_ ? lctx_->Spawn() : nullptr;
  auto new_sctx = std::make_unique<SPUContext>(config_, new_lctx);
  new_sctx->prot_ = prot_->fork();
  return new_sctx;
}

void setupTrace(spu::SPUContext* sctx, const spu::RuntimeConfig& rt_config) {
  int64_t tr_flag = 0;
  // TODO: Support tracing for parallel op execution
  if (rt_config.enable_action_trace() &&
      !rt_config.experimental_enable_intra_op_par()) {
    tr_flag |= TR_LOG;
  }

  if (rt_config.enable_pphlo_profile()) {
    tr_flag |= TR_HLO;
    tr_flag |= TR_REC;
  }

  if (rt_config.enable_hal_profile()) {
    tr_flag |= TR_HAL | TR_MPC;
    tr_flag |= TR_REC;
  }

  initTrace(sctx->id(), tr_flag);
  GET_TRACER(sctx)->getProfState()->clearRecords();
}

}  // namespace spu
