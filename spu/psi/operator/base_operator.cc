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

#include "spu/psi/operator/base_operator.h"

#include "spdlog/spdlog.h"

#include "spu/psi/utils/serialize.h"

namespace spu::psi {

PsiBaseOperator::PsiBaseOperator(
    const std::shared_ptr<yasl::link::Context>& link_ctx)
    : link_ctx_(link_ctx) {}

std::vector<std::string> PsiBaseOperator::Run(
    const std::vector<std::string>& inputs, bool broadcast_result) {
  auto res = OnRun(inputs);

  if (broadcast_result) {
    size_t max_size = res.size();
    size_t broadcast_rank = 0;
    std::vector<size_t> res_size_list =
        AllGatherItemsSize(link_ctx_, res.size());
    for (size_t i = 0; i < res_size_list.size(); ++i) {
      max_size = std::max(max_size, res_size_list[i]);
      if (res_size_list[i] > 0) {
        // in broadcast case, there should be only one party have results
        YASL_ENFORCE(broadcast_rank == 0);
        broadcast_rank = i;
      }
    }
    if (max_size == 0) {
      // no need broadcast
      return res;
    }
    auto recv_res_buf =
        yasl::link::Broadcast(link_ctx_, utils::SerializeStrItems(res),
                              broadcast_rank, "broadcast psi result");
    if (res.empty()) {
      // use broadcast result
      utils::DeserializeStrItems(recv_res_buf, &res);
    }
  }

  return res;
}

std::vector<size_t> AllGatherItemsSize(
    const std::shared_ptr<yasl::link::Context>& link_ctx, size_t self_size) {
  std::vector<size_t> items_size_list(link_ctx->WorldSize());

  std::vector<yasl::Buffer> items_size_buf_list = yasl::link::AllGather(
      link_ctx, utils::SerializeSize(self_size), "PSI:SYNC_SIZE");

  for (size_t idx = 0; idx < items_size_buf_list.size(); idx++) {
    items_size_list[idx] = utils::DeserializeSize(items_size_buf_list[idx]);
  }

  return items_size_list;
}

}  // namespace spu::psi
