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

#include "spu/psi/core/communication.h"

#include "spdlog/spdlog.h"

namespace spu::psi {

std::shared_ptr<yasl::link::Context> CreateP2PLinkCtx(
    const std::string& id_prefix,
    const std::shared_ptr<yasl::link::Context>& link_ctx, size_t peer_rank) {
  if (link_ctx->WorldSize() > 2) {
    // build subworld link
    auto peer_id = link_ctx->PartyIdByRank(peer_rank);
    auto self_id = link_ctx->PartyIdByRank(link_ctx->Rank());
    std::vector<std::string> party_ids({peer_id, self_id});
    std::sort(party_ids.begin(), party_ids.end());

    size_t a_rank = std::min(link_ctx->Rank(), peer_rank);
    size_t b_rank = std::max(link_ctx->Rank(), peer_rank);
    std::string sub_id = fmt::format("{}-{}-{}", id_prefix, a_rank, b_rank);

    auto ctx = link_ctx->SubWorld(sub_id, party_ids);
    SPDLOG_INFO("create p2p link, id:{}, rank:{}", ctx->Id(), ctx->Rank());

    return ctx;
  } else {
    return link_ctx;
  }
}

}  // namespace spu::psi
