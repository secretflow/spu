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

#include "spu/psi/operator/nparty_psi.h"

#include <algorithm>
#include <future>
#include <memory>
#include <utility>

#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/utils/parallel.h"

#include "spu/psi/core/communication.h"
#include "spu/psi/operator/kkrt_2party_psi.h"
#include "spu/psi/utils/serialize.h"

namespace spu::psi {

namespace {
constexpr size_t kSyncRecvWaitTimeoutMs = 60 * 60 * 1000;
}

NpartyPsiOperator::NpartyPsiOperator(const Options& options)
    : PsiBaseOperator(options.link_ctx), options_(options) {
  YASL_ENFORCE(options_.link_ctx->WorldSize() >= 2);
}

std::vector<std::string> NpartyPsiOperator::OnRun(
    const std::vector<std::string>& inputs) {
  std::vector<std::pair<size_t, size_t>> party_size_rank_vec =
      GetAllPartyItemSizeVec(inputs.size());
  if ((party_size_rank_vec[0].first == 0) ||
      (party_size_rank_vec[1].first == 0)) {
    // check master and min item size whether is zero
    return {};
  }

  // std::log2(absl::bit_ceil( 00000000 )) = 0
  // std::log2(absl::bit_ceil( 00000001 )) = 0
  // std::log2(absl::bit_ceil( 00000010 )) = 1
  // std::log2(absl::bit_ceil( 00000011 )) = 2
  // std::log2(absl::bit_ceil( 00000100 )) = 2
  // std::log2(absl::bit_ceil( 00000101 )) = 3
  // std::log2(absl::bit_ceil( 00000110 )) = 3
  // std::log2(absl::bit_ceil( 00000111 )) = 3
  // std::log2(absl::bit_ceil( 00001000 )) = 3
  // std::log2(absl::bit_ceil( 00001001 )) = 4
  size_t level_num = std::log2(absl::bit_ceil(party_size_rank_vec.size()));

  std::vector<std::string> intersection;
  for (size_t li = 0; li < level_num; ++li) {
    size_t peer_rank, target_rank;
    GetPsiRank(party_size_rank_vec, &peer_rank, &target_rank);
    if (li == 0) {
      intersection = Run2PartyPsi(inputs, peer_rank, target_rank);
    } else {
      intersection = Run2PartyPsi(intersection, peer_rank, target_rank);
    }

    SPDLOG_INFO("rank:{}, level_idx:{}, level_num:{}, intersection:{}",
                options_.link_ctx->Rank(), li, level_num, intersection.size());

    // erase non-target rank
    size_t erase_pos = (party_size_rank_vec.size() + 1) / 2;
    for (size_t idx = erase_pos; idx < party_size_rank_vec.size(); ++idx) {
      if (options_.link_ctx->Rank() == party_size_rank_vec[idx].second) {
        yasl::link::RecvTimeoutGuard guard(options_.link_ctx,
                                           kSyncRecvWaitTimeoutMs);
        auto recv_intersection_buf = yasl::link::Broadcast(
            options_.link_ctx, {}, options_.master_rank, "recv finish message");
        return {};
      }
    }
    party_size_rank_vec.erase(party_size_rank_vec.begin() + erase_pos,
                              party_size_rank_vec.end());

    // gather all intersection size
    std::vector<std::string> sub_party_ids(party_size_rank_vec.size());
    for (size_t idx = 0; idx < party_size_rank_vec.size(); ++idx) {
      sub_party_ids[idx] =
          options_.link_ctx->PartyIdByRank(party_size_rank_vec[idx].second);
    }
    std::string sub_id =
        fmt::format("subid-level:{}-{}", li, party_size_rank_vec.size());
    std::shared_ptr<yasl::link::Context> sub_link_ctx =
        options_.link_ctx->SubWorld(sub_id, sub_party_ids);
    std::vector<yasl::Buffer> gather_size_bufs = yasl::link::AllGather(
        sub_link_ctx, utils::SerializeSize(intersection.size()),
        fmt::format("round:{}, {} gather item size", li, sub_link_ctx->Rank()));

    size_t min_intersection_size = inputs.size();
    for (size_t idx = 0; idx < gather_size_bufs.size(); ++idx) {
      size_t current_idx_size = utils::DeserializeSize(gather_size_bufs[idx]);
      min_intersection_size = std::min(min_intersection_size, current_idx_size);
      if (min_intersection_size == 0) {
        break;
      }
    }

    // check current loop has zero size intersection,
    if (min_intersection_size == 0) {
      intersection.resize(0);
      if (options_.link_ctx->Rank() == options_.master_rank) {
        yasl::link::Broadcast(options_.link_ctx, "finish", options_.master_rank,
                              "send finish message");
      } else {
        yasl::link::Broadcast(options_.link_ctx, {}, options_.master_rank,
                              "recv finish message");
      }
      return intersection;
    }

    // broadcast intersection
    if (party_size_rank_vec.size() == 1) {
      yasl::link::Broadcast(options_.link_ctx, "finish", options_.master_rank,
                            "send finish message");
      std::sort(intersection.begin(), intersection.end());
      return intersection;
    }
  }

  return {};
}

std::vector<std::string> NpartyPsiOperator::Run2PartyPsi(
    const std::vector<std::string>& items, size_t peer_rank,
    size_t target_rank) {
  SPDLOG_INFO("Run2PartyPsi:{}, peer_rank:{}, target_rank:{}, item_size:{}",
              options_.link_ctx->Rank(), peer_rank, target_rank, items.size());
  if (peer_rank == options_.link_ctx->Rank()) {
    return items;
  }

  auto link_ctx = CreateP2PLinkCtx("2partypsi", options_.link_ctx, peer_rank);

  if (options_.psi_type == PsiType::Ecdh) {
    return RunEcdhPsi(link_ctx, items,
                      target_rank == options_.link_ctx->Rank()
                          ? link_ctx->Rank()
                          : link_ctx->NextRank(),
                      options_.curve_type, options_.batch_size);
  } else if (options_.psi_type == PsiType::Kkrt) {
    KkrtPsiOperator::Options opts;
    opts.link_ctx = link_ctx;
    opts.receiver_rank = target_rank == options_.link_ctx->Rank()
                             ? opts.link_ctx->Rank()
                             : opts.link_ctx->NextRank();
    KkrtPsiOperator kkrt_op(opts);

    return kkrt_op.Run(items, false);
  } else {
    YASL_THROW("not support psi type: {}", static_cast<int>(options_.psi_type));
  }
}

std::vector<std::pair<size_t, size_t>>
NpartyPsiOperator::GetAllPartyItemSizeVec(size_t item_size) {
  // get all party's item size
  std::vector<std::pair<size_t, size_t>> party_size_rank_vec;

  std::vector<yasl::Buffer> gather_size = yasl::link::AllGather(
      options_.link_ctx, utils::SerializeSize(item_size),
      fmt::format("{} send item size", options_.link_ctx->Rank()));
  YASL_ENFORCE(gather_size.size() == options_.link_ctx->WorldSize());

  for (size_t idx = 0; idx < options_.link_ctx->WorldSize(); ++idx) {
    size_t idx_item_size = utils::DeserializeSize(gather_size[idx]);

    party_size_rank_vec.emplace_back(idx_item_size, idx);
  }
  if (options_.master_rank != 0) {
    // place master to first
    std::swap(party_size_rank_vec[0],
              party_size_rank_vec[options_.master_rank]);
  }
  // ascending sort other rank by items size,
  std::sort(party_size_rank_vec.begin() + 1, party_size_rank_vec.end());

  return party_size_rank_vec;
}

void NpartyPsiOperator::GetPsiRank(
    const std::vector<std::pair<size_t, size_t>>& party_size_rank_vec,
    size_t* peer_rank, size_t* target_rank) {
  if ((party_size_rank_vec.size() % 2 != 0) &&
      (party_size_rank_vec[party_size_rank_vec.size() / 2].second ==
       options_.link_ctx->Rank())) {
    // no peer
    *peer_rank = options_.link_ctx->Rank();
    *target_rank = options_.link_ctx->Rank();
    return;
  }

  for (size_t idx = 0; idx < party_size_rank_vec.size() / 2; ++idx) {
    // peer_idx begin from bottom
    size_t peer_idx = party_size_rank_vec.size() - 1 - idx;
    if (party_size_rank_vec[idx].second == options_.link_ctx->Rank()) {
      *peer_rank = party_size_rank_vec[peer_idx].second;
      *target_rank = party_size_rank_vec[idx].second;

      return;
    } else if (party_size_rank_vec[peer_idx].second ==
               options_.link_ctx->Rank()) {
      *peer_rank = party_size_rank_vec[idx].second;
      *target_rank = party_size_rank_vec[idx].second;

      return;
    }
  }

  YASL_THROW("can not find self rank({}) in party_size_rank_vec",
             options_.link_ctx->Rank());
}
}  // namespace spu::psi
