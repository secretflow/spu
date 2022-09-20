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

#include "spu/psi/core/ecdh_psi.h"

#include <future>
#include <utility>

#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/utils/parallel.h"
#include "yasl/utils/serialize.h"

#include "spu/psi/core/communication.h"
#include "spu/psi/cryptor/cryptor_selector.h"
#include "spu/psi/utils/batch_provider.h"

namespace spu::psi {

EcdhPsiContext::EcdhPsiContext(const EcdhPsiOptions& options)
    : options_(options) {
  YASL_ENFORCE(options_.link_ctx->WorldSize() == 2);

  main_link_ctx_ = options_.link_ctx;
  dual_mask_link_ctx_ = options_.link_ctx->Spawn();
}

void EcdhPsiContext::CheckConfig() {
  // Sanity check: the `target_rank` and 'curve_type' should match.
  std::string my_config =
      fmt::format("target_rank={},curve={}", options_.target_rank,
                  static_cast<int>(options_.ecc_cryptor->GetCurveType()));
  yasl::Buffer my_config_buf(my_config.c_str(), my_config.size());
  auto config_list =
      yasl::link::AllGather(main_link_ctx_, my_config_buf, "ECDHPSI:SANITY");
  auto peer_config = config_list[main_link_ctx_->NextRank()];
  YASL_ENFORCE(my_config_buf == peer_config,
               "EcdhPsiContext Config mismatch, mine={}, peer={}", my_config,
               peer_config);
}

void EcdhPsiContext::MaskSelf(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
  size_t batch_count = 0;
  while (true) {
    // NOTE: we still need to send one batch even there is no data.
    // This dummy batch is used to notify peer the end of data stream.
    auto batch_items = batch_provider->ReadNextBatch(options_.batch_size);
    std::vector<std::string> masked_items;
    if (!batch_items.empty()) {
      masked_items = Mask(options_.ecc_cryptor,
                          HashInputs(options_.ecc_cryptor, batch_items));
    }
    // Send x^a.
    const auto tag = fmt::format("ECDHPSI:X^A:{}", batch_count);
    SendBatch(masked_items, tag);
    if (batch_items.empty()) {
      SPDLOG_INFO("MaskSelf:{}--finished, batch_count={}",
                  options_.link_ctx->Id(), batch_count);
      break;
    }
    ++batch_count;
  }
}

void EcdhPsiContext::MaskPeer(
    const std::shared_ptr<ICipherStore>& cipher_store) {
  size_t batch_count = 0;
  while (true) {
    // Fetch y^b.
    std::vector<std::string> peer_items;
    std::vector<std::string> dual_masked_peers;
    const auto tag = fmt::format("ECDHPSI:Y^B:{}", batch_count);
    RecvBatch(&peer_items, tag);

    // Compute (y^b)^a.
    if (!peer_items.empty()) {
      // TODO: avoid mem copy
      for (const auto& masked : Mask(options_.ecc_cryptor, peer_items)) {
        // In the final comparison, we only send & compare `kFinalCompareBytes`
        // number of bytes.
        std::string cipher = masked.substr(
            masked.length() - options_.dual_mask_size, options_.dual_mask_size);
        if (CanTouchResults()) {
          // Store cipher of peer items for later intersection compute.
          cipher_store->SavePeer(cipher);
        }
        dual_masked_peers.emplace_back(std::move(cipher));
      }
    }
    // Should send out the dual masked items to peer.
    if (PeerCanTouchResults()) {
      const auto tag = fmt::format("ECDHPSI:Y^B^A:{}", batch_count);
      SendDualMaskedBatch(dual_masked_peers, tag);
    }
    if (peer_items.empty()) {
      SPDLOG_INFO("MaskPeer:{}--finished, batch_count={}",
                  options_.link_ctx->Id(), batch_count);
      break;
    }
    batch_count++;
  }
}

void EcdhPsiContext::RecvDualMaskedSelf(
    const std::shared_ptr<ICipherStore>& cipher_store) {
  if (!CanTouchResults()) {
    return;
  }

  // Receive x^a^b.
  size_t batch_count = 0;
  while (true) {
    // TODO: avoid mem copy
    std::vector<std::string> masked_items;
    const auto tag = fmt::format("ECDHPSI:X^A^B:{}", batch_count);
    RecvDualMaskedBatch(&masked_items, tag);
    for (auto& item : masked_items) {
      cipher_store->SaveSelf(std::move(item));
    }
    if (masked_items.empty()) {
      SPDLOG_INFO("{} recv last batch finished, batch_count={}",
                  options_.link_ctx->Id(), batch_count);
      break;
    }
    batch_count++;
  }
}

void EcdhPsiContext::SendBatch(const std::vector<std::string>& batch_items,
                               std::string_view tag) {
  SendBatchImpl(batch_items, main_link_ctx_, tag);
}

void EcdhPsiContext::SendBatch(const std::vector<std::string_view>& batch_items,
                               std::string_view tag) {
  SendBatchImpl(batch_items, main_link_ctx_, tag);
}

void EcdhPsiContext::RecvBatch(std::vector<std::string>* items,
                               std::string_view tag) {
  RecvBatchImpl(main_link_ctx_, items, tag);
}

void EcdhPsiContext::SendDualMaskedBatch(
    const std::vector<std::string>& batch_items, std::string_view tag) {
  SendBatchImpl(batch_items, dual_mask_link_ctx_, tag);
}

void EcdhPsiContext::SendDualMaskedBatch(
    const std::vector<std::string_view>& batch_items, std::string_view tag) {
  SendBatchImpl(batch_items, dual_mask_link_ctx_, tag);
}

void EcdhPsiContext::RecvDualMaskedBatch(std::vector<std::string>* items,
                                         std::string_view tag) {
  RecvBatchImpl(dual_mask_link_ctx_, items, tag);
}

void EcdhPsiContext::SendBatchImpl(
    const std::vector<std::string>& batch_items,
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    std::string_view tag) {
  PsiDataBatch batch;
  batch.is_last_batch = batch_items.empty();
  batch.item_num = batch_items.size();
  // Mask and Send this batch.
  if (!batch_items.empty()) {
    batch.flatten_bytes.reserve(batch_items.size() * batch_items[0].size());
    for (const auto& item : batch_items) {
      batch.flatten_bytes.append(item);
    }
  }

  link_ctx->SendAsync(link_ctx->NextRank(), batch.Serialize(), tag);
}

void EcdhPsiContext::SendBatchImpl(
    const std::vector<std::string_view>& batch_items,
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    std::string_view tag) {
  PsiDataBatch batch;
  batch.is_last_batch = batch_items.empty();
  batch.item_num = batch_items.size();
  // Mask and Send this batch.
  if (!batch_items.empty()) {
    batch.flatten_bytes.reserve(batch_items.size() * batch_items[0].size());
    for (const auto& item : batch_items) {
      batch.flatten_bytes.append(item);
    }
  }

  link_ctx->SendAsync(link_ctx->NextRank(), batch.Serialize(), tag);
}

void EcdhPsiContext::RecvBatchImpl(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    std::vector<std::string>* items, std::string_view tag) {
  PsiDataBatch batch =
      PsiDataBatch::Deserialize(link_ctx->Recv(link_ctx->NextRank(), tag));

  if (batch.item_num > 0) {
    auto item_size = batch.flatten_bytes.size() / batch.item_num;
    for (size_t i = 0; i < batch.item_num; ++i) {
      items->emplace_back(batch.flatten_bytes.substr(i * item_size, item_size));
    }
  }
}

void RunEcdhPsi(const EcdhPsiOptions& options,
                const std::shared_ptr<IBatchProvider>& batch_provider,
                const std::shared_ptr<ICipherStore>& cipher_store) {
  YASL_ENFORCE(options.link_ctx->WorldSize() == 2);
  YASL_ENFORCE(batch_provider != nullptr && cipher_store != nullptr);

  EcdhPsiContext handler(options);
  handler.CheckConfig();

  std::future<void> f_mask_self =
      std::async([&] { return handler.MaskSelf(batch_provider); });
  std::future<void> f_mask_peer =
      std::async([&] { return handler.MaskPeer(cipher_store); });
  std::future<void> f_recv_peer =
      std::async([&] { return handler.RecvDualMaskedSelf(cipher_store); });

  // Wait for end of logic flows or exceptions.
  // Note: exception_ptr is `shared_ptr` style, hence could be used to prolong
  // the lifetime of pointed exceptions.
  std::exception_ptr mask_self_exptr = nullptr;
  std::exception_ptr mask_peer_exptr = nullptr;
  std::exception_ptr recv_peer_exptr = nullptr;

  try {
    f_mask_self.get();
  } catch (const std::exception& e) {
    mask_self_exptr = std::current_exception();
    SPDLOG_ERROR("Error in mask self: {}", e.what());
  }

  try {
    f_mask_peer.get();
  } catch (const std::exception& e) {
    mask_peer_exptr = std::current_exception();
    SPDLOG_ERROR("Error in mask peer: {}", e.what());
  }

  try {
    f_recv_peer.get();
  } catch (const std::exception& e) {
    recv_peer_exptr = std::current_exception();
    SPDLOG_ERROR("Error in recv peer: {}", e.what());
  }

  if (mask_self_exptr) {
    std::rethrow_exception(mask_self_exptr);
  }
  if (mask_peer_exptr) {
    std::rethrow_exception(mask_peer_exptr);
  }
  if (recv_peer_exptr) {
    std::rethrow_exception(recv_peer_exptr);
  }
}

std::vector<std::string> RunEcdhPsi(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const std::vector<std::string>& items, size_t target_rank, CurveType curve,
    size_t batch_size) {
  EcdhPsiOptions options;
  options.ecc_cryptor = CreateEccCryptor(curve);
  options.link_ctx = link_ctx;
  options.target_rank = target_rank;
  options.batch_size = batch_size;

  auto memory_store = std::make_shared<MemoryCipherStore>();
  auto batch_provider = std::make_shared<MemoryBatchProvider>(items);

  RunEcdhPsi(options, batch_provider, memory_store);

  // Originally we should setup a hashset for peer results.
  // But tests show that when items_count > 10,000,000, the performance of
  // |std::unordered_set| or |absl::flat_hash_set| drops significantly.
  // Besides, these hashset containers require more memory.
  // Here we choose the compact data structure and stable find costs.
  std::vector<std::string> ret;
  std::vector<std::string> peer_results(memory_store->peer_results());
  std::sort(peer_results.begin(), peer_results.end());
  const auto& self_results = memory_store->self_results();
  for (uint32_t index = 0; index < self_results.size(); index++) {
    if (std::binary_search(peer_results.begin(), peer_results.end(),
                           self_results[index])) {
      YASL_ENFORCE(index < items.size());
      ret.push_back(items[index]);
    }
  }
  return ret;
}

}  // namespace spu::psi
