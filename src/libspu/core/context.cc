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

#include "yacl/link/algorithm/allgather.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/config.h"
#include "libspu/core/trace.h"

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
      lctx_(lctx),
      max_cluster_level_concurrency_(yacl::get_num_threads()) {
  populateRuntimeConfig(config_);
  // Limit number of threads
  if (config.max_concurrency > 0) {
    yacl::set_num_threads(config.max_concurrency);
    max_cluster_level_concurrency_ = std::min<int32_t>(
        max_cluster_level_concurrency_, config.max_concurrency);
  }

  if (lctx_) {
    auto other_max = yacl::link::AllGather(
        lctx, {&max_cluster_level_concurrency_, sizeof(int32_t)}, "num_cores");

    // Comupte min
    for (const auto& o : other_max) {
      max_cluster_level_concurrency_ = std::min<int32_t>(
          max_cluster_level_concurrency_, o.data<int32_t>()[0]);
    }
  }
}

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
  if (rt_config.enable_action_trace &&
      !rt_config.experimental_enable_intra_op_par) {
    tr_flag |= TR_LOG;
  }

  if (rt_config.enable_pphlo_profile) {
    tr_flag |= TR_HLO;
    tr_flag |= TR_REC;
  }

  if (rt_config.enable_hal_profile) {
    tr_flag |= TR_HAL | TR_MPC;
    tr_flag |= TR_REC;
  }

  initTrace(sctx->id(), tr_flag);
  GET_TRACER(sctx)->getProfState()->clearRecords();
}

}  // namespace spu
