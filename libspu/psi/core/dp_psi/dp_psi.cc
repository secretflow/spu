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

#include "libspu/psi/core/dp_psi/dp_psi.h"

#include <algorithm>
#include <future>
#include <random>

#include "spdlog/spdlog.h"
#include "yacl/base/buffer.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/communication.h"
#include "libspu/psi/core/dp_psi/dp_psi_utils.h"
#include "libspu/psi/core/ecdh_3pc_psi.h"
#include "libspu/psi/cryptor/cryptor_selector.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/cipher_store.h"
#include "libspu/psi/utils/serialize.h"

#include "libspu/psi/utils/serializable.pb.h"

namespace spu::psi {

namespace {

constexpr uint64_t kSendBatchSize = 8192;

std::vector<size_t> BernoulliSamples(const std::vector<size_t>& items_idx,
                                     double q) {
  std::pair<uint64_t, uint64_t> seed_pair =
      yacl::DecomposeUInt128(yacl::crypto::RandU128());
  std::mt19937 rand(seed_pair.first);

  SPDLOG_INFO("sample bernoulli_distribution: {}", q);

  std::bernoulli_distribution dist(q);

  std::vector<size_t> bernoulli_items_idx;
  for (unsigned long idx : items_idx) {
    if (dist(rand)) {
      bernoulli_items_idx.push_back(idx);
    }
  }

  SPDLOG_INFO(
      "bernoulli_items_idx:{} ratio:{} ", bernoulli_items_idx.size(),
      static_cast<double>(bernoulli_items_idx.size()) / items_idx.size());

  return bernoulli_items_idx;
}

std::pair<std::vector<std::string>, std::vector<size_t>> BernoulliSamples(
    const std::vector<std::string>& items,
    const std::vector<size_t>& shuffled_idx, double p) {
  std::pair<uint64_t, uint64_t> seed_pair =
      yacl::DecomposeUInt128(yacl::crypto::RandU128());
  std::mt19937 rand(seed_pair.first);

  SPDLOG_INFO("sample bernoulli_distribution: {}", p);

  std::bernoulli_distribution dist(p);

  std::vector<std::string> bernoulli_items;
  std::vector<size_t> bernoulli_idx;
  for (size_t idx = 0; idx < items.size(); idx++) {
    if (dist(rand)) {
      bernoulli_items.push_back(items[shuffled_idx[idx]]);
      bernoulli_idx.push_back(shuffled_idx[idx]);
    }
  }

  SPDLOG_INFO("bernoulli_items: {}, bernoulli_idx:{} ratio:{} ",
              bernoulli_items.size(), bernoulli_idx.size(),
              static_cast<double>(bernoulli_items.size()) / items.size());
  return std::make_pair(bernoulli_items, bernoulli_idx);
}

std::vector<size_t> GetShuffledIdx(size_t items_size) {
  std::pair<uint64_t, uint64_t> seed_pair =
      yacl::DecomposeUInt128(yacl::crypto::RandU128());
  std::mt19937 rng(seed_pair.first);

  std::vector<size_t> shuffled_idx_vec(items_size);
  std::iota(shuffled_idx_vec.begin(), shuffled_idx_vec.end(), 0);
  std::shuffle(shuffled_idx_vec.begin(), shuffled_idx_vec.end(), rng);

  return shuffled_idx_vec;
}

}  // namespace

size_t RunDpEcdhPsiAlice(const DpPsiOptions& dp_psi_options,
                         const std::shared_ptr<yacl::link::Context>& link_ctx,
                         const std::vector<std::string>& items,
                         size_t* sub_sample_size, size_t* up_sample_size,
                         CurveType curve) {
  auto memory_store = std::make_shared<MemoryCipherStore>();
  auto batch_provider = std::make_shared<MemoryBatchProvider>(items);

  SPDLOG_INFO(
      "alice items_size: {}, down_sampling_rate: {}, up_sampling_rate: {}",
      items.size(), dp_psi_options.p2, dp_psi_options.q);

  EcdhPsiOptions options;

  options.link_ctx = link_ctx;
  options.ecc_cryptor = CreateEccCryptor(curve);
  options.target_rank = link_ctx->Rank();

  EcdhP2PExtendCtx psi_ctx(options);

  std::future<void> f_mask_self_a =
      std::async([&] { return psi_ctx.MaskSelf(batch_provider); });

  std::future<void> f_recv_peer_a =
      std::async([&] { return psi_ctx.MaskPeer(memory_store); });

  f_mask_self_a.get();
  f_recv_peer_a.get();

  SPDLOG_INFO("after maskSelf maskPeer");

  std::vector<std::string>& alice_peer_result = memory_store->peer_results();

  std::vector<std::string> self_dual_mask;

  std::future<void> f_recv_dual_mask_a =
      std::async([&] { return psi_ctx.RecvItems(&self_dual_mask); });
  f_recv_dual_mask_a.get();

  // std::vector<std::string> intersection_ab;
  std::vector<size_t> intersection_idx;
  std::vector<size_t> non_intersection_idx;

  std::sort(self_dual_mask.begin(), self_dual_mask.end());

  for (size_t index = 0; index < alice_peer_result.size(); index++) {
    if (std::binary_search(self_dual_mask.begin(), self_dual_mask.end(),
                           alice_peer_result[index])) {
      SPU_ENFORCE(index < alice_peer_result.size());

      intersection_idx.push_back(index);
    } else {
      non_intersection_idx.push_back(index);
    }
  }
  // check non_intersection_idx size==0
  // if size==0 report intersection 0
  if (non_intersection_idx.empty()) {
    yacl::Buffer intersection_idx_size_buffer = utils::SerializeSize(0);
    link_ctx->SendAsyncThrottled(link_ctx->NextRank(),
                                 intersection_idx_size_buffer,
                                 fmt::format("intersection_idx size: {}", 0));

    SPDLOG_WARN("non_intersection_idx size 0");

    return 0;
  }

  std::vector<size_t> sub_sample_idx =
      BernoulliSamples(intersection_idx, dp_psi_options.p2);

  std::vector<size_t> up_sample_idx =
      BernoulliSamples(non_intersection_idx, dp_psi_options.q);

  *sub_sample_size = intersection_idx.size() - sub_sample_idx.size();
  *up_sample_size = up_sample_idx.size();

  sub_sample_idx.insert(sub_sample_idx.end(), up_sample_idx.begin(),
                        up_sample_idx.end());
  // sort index, mix sub_sample_idx and up_sample_idx
  std::sort(sub_sample_idx.begin(), sub_sample_idx.end());

  SPDLOG_INFO("alice intersection size: {}", sub_sample_idx.size());

  yacl::Buffer intersection_idx_size_buffer =
      utils::SerializeSize(sub_sample_idx.size());
  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), intersection_idx_size_buffer,
      fmt::format("intersection_idx size: {}", sub_sample_idx.size()));

  for (size_t idx = 0; idx < sub_sample_idx.size(); idx += kSendBatchSize) {
    PsiDataBatch data_batch;
    size_t current_batch_size;
    if ((idx + kSendBatchSize) < sub_sample_idx.size()) {
      data_batch.is_last_batch = false;
      current_batch_size = kSendBatchSize;
    } else {
      data_batch.is_last_batch = true;
      current_batch_size = sub_sample_idx.size() - idx;
    }
    std::string flatten_bytes(current_batch_size * sizeof(size_t), '\0');

    std::memcpy(flatten_bytes.data(), &sub_sample_idx[idx],
                current_batch_size * sizeof(size_t));

    data_batch.flatten_bytes = flatten_bytes;
    link_ctx->SendAsyncThrottled(link_ctx->NextRank(), data_batch.Serialize(),
                                 "batch send idx");
  }

  return sub_sample_idx.size();
}

std::vector<size_t> RunDpEcdhPsiBob(
    const DpPsiOptions& dp_psi_options,
    const std::shared_ptr<yacl::link::Context>& link_ctx,
    const std::vector<std::string>& items, size_t* sub_sample_size,
    CurveType curve) {
  std::vector<size_t> bob_shuffled_idx = GetShuffledIdx(items.size());

  SPDLOG_INFO("bob items_size: {}, down_sampling_rate: {}", items.size(),
              dp_psi_options.p1);

  std::pair<std::vector<std::string>, std::vector<size_t>> sub_sample_result =
      BernoulliSamples(items, bob_shuffled_idx, dp_psi_options.p1);

  *sub_sample_size = items.size() - sub_sample_result.first.size();

  auto batch_provider =
      std::make_shared<MemoryBatchProvider>(sub_sample_result.first);
  auto memory_store = std::make_shared<MemoryCipherStore>();

  EcdhPsiOptions options;

  // set options
  options.ecc_cryptor = CreateEccCryptor(curve);
  options.link_ctx = link_ctx;
  options.target_rank = link_ctx->Rank();

  EcdhP2PExtendCtx psi_ctx(options);

  std::future<void> f_mask_peer_b =
      std::async([&] { return psi_ctx.MaskPeer(memory_store); });

  std::future<void> f_mask_self_b =
      std::async([&] { return psi_ctx.MaskSelf(batch_provider); });

  f_mask_peer_b.get();
  f_mask_self_b.get();

  SPDLOG_INFO("after mask self, mask peer");

  std::vector<std::string>& bob_peer_result = memory_store->peer_results();
  std::sort(bob_peer_result.begin(), bob_peer_result.end());

  // send x^a^b to alice
  std::future<void> f_bob_send_shuffle_dual_mask =
      std::async([&] { return psi_ctx.SendItems(bob_peer_result); });

  f_bob_send_shuffle_dual_mask.get();

  SPDLOG_INFO("after send shuffled batch");

  yacl::Buffer intersection_size_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv intersection_size"));
  size_t intersection_size = utils::DeserializeSize(intersection_size_buffer);

  std::vector<size_t> intersection_idx(intersection_size);
  if (intersection_size == 0) {
    return intersection_idx;
  }

  size_t recv_idx = 0;
  while (true) {
    PsiDataBatch batch = PsiDataBatch::Deserialize(link_ctx->Recv(
        link_ctx->NextRank(), fmt::format("recv batch idx{}", recv_idx)));

    SPU_ENFORCE(batch.flatten_bytes.size() % sizeof(size_t) == 0);
    size_t current_num;
    current_num = batch.flatten_bytes.size() / sizeof(size_t);
    std::memcpy(intersection_idx.data() + recv_idx, batch.flatten_bytes.data(),
                batch.flatten_bytes.size());

    recv_idx += current_num;

    if (batch.is_last_batch) {
      SPDLOG_INFO("recv last batch, recv_num: {}", recv_idx);
      break;
    }
  }

  std::vector<size_t> dp_intersection_idx;
  dp_intersection_idx.reserve(intersection_idx.size());
  for (const auto& idx : intersection_idx) {
    dp_intersection_idx.push_back(sub_sample_result.second[idx]);
  }
  std::sort(dp_intersection_idx.begin(), dp_intersection_idx.end());

  SPDLOG_INFO("dp_intersection_idx size:{}", dp_intersection_idx.size());

  return dp_intersection_idx;
}

}  // namespace spu::psi
