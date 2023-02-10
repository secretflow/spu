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

#include "libspu/psi/operator/base_operator.h"

#include <utility>

#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

namespace spu::psi {

PsiBaseOperator::PsiBaseOperator(std::shared_ptr<yacl::link::Context> link_ctx)
    : link_ctx_(std::move(link_ctx)) {}

std::vector<std::string> PsiBaseOperator::Run(
    const std::vector<std::string>& inputs, bool broadcast_result) {
  auto run_f = std::async([&] { return OnRun(inputs); });
  auto res = SyncWait(link_ctx_, &run_f);

  if (broadcast_result) {
    size_t max_size = res.size();
    size_t broadcast_rank = 0;
    std::vector<size_t> res_size_list =
        AllGatherItemsSize(link_ctx_, res.size());
    for (size_t i = 0; i < res_size_list.size(); ++i) {
      max_size = std::max(max_size, res_size_list[i]);
      if (res_size_list[i] > 0) {
        // in broadcast case, there should be only one party have results
        SPU_ENFORCE(broadcast_rank == 0);
        broadcast_rank = i;
      }
    }
    if (max_size == 0) {
      // no need broadcast
      return res;
    }
    auto recv_res_buf =
        yacl::link::Broadcast(link_ctx_, utils::SerializeStrItems(res),
                              broadcast_rank, "broadcast psi result");
    if (res.empty()) {
      // use broadcast result
      utils::DeserializeStrItems(recv_res_buf, &res);
    }
  }

  return res;
}

}  // namespace spu::psi
