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

#include "libspu/psi/core/ecdh_3pc_psi.h"

#include <algorithm>
#include <future>
#include <random>
#include <utility>

#include "fmt/format.h"
#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"
#include "yacl/utils/serialize.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/cryptor/cryptor_selector.h"

namespace spu::psi {

EcdhP2PExtendCtx::EcdhP2PExtendCtx(const EcdhPsiOptions& options)
    : EcdhPsiContext(options) {}

void EcdhP2PExtendCtx::MaskSendSelf(
    const std::vector<std::string>& self_items) {
  auto batch_provider = std::make_shared<MemoryBatchProvider>(self_items);

  MaskSelf(batch_provider);
}

void EcdhP2PExtendCtx::MaskRecvPeer(
    std::vector<std::string>* dup_masked_peer_items) {
  auto memory_store = std::make_shared<MemoryCipherStore>();

  MaskPeer(memory_store);

  *dup_masked_peer_items = memory_store->peer_results();
}

void EcdhP2PExtendCtx::MaskShufflePeer() {
  std::vector<std::string> peer_items;
  RecvItems(&peer_items);

  std::vector<std::string> dup_masked_items;
  if (!peer_items.empty()) {
    for (const auto& masked : Mask(options_.ecc_cryptor, peer_items)) {
      dup_masked_items.emplace_back(masked.substr(
          masked.length() - options_.dual_mask_size, options_.dual_mask_size));
    }
    // shuffle x^a^b
    std::sort(dup_masked_items.begin(), dup_masked_items.end());
  }

  SendDupMasked(dup_masked_items);
}

void EcdhP2PExtendCtx::MaskPeerForward(
    const std::shared_ptr<EcdhP2PExtendCtx>& forward_ctx,
    int32_t truncation_size) {
  size_t batch_count = 0;
  while (true) {
    std::vector<std::string> peer_items;
    std::vector<std::string> dup_masked_items;
    RecvBatch(&peer_items, batch_count);
    if (!peer_items.empty()) {
      for (auto& masked : Mask(options_.ecc_cryptor, peer_items)) {
        if (truncation_size > 0) {
          dup_masked_items.emplace_back(masked.substr(
              masked.length() - truncation_size, truncation_size));
        } else {
          dup_masked_items.emplace_back(std::move(masked));
        }
      }
    }
    forward_ctx->ForwardBatch(dup_masked_items, batch_count);
    if (peer_items.empty()) {
      SPDLOG_INFO("MaskPeerForward:{} finished, batch_count={}",
                  options_.link_ctx->Id(), batch_count);
      break;
    }
    batch_count++;
  }
}

void EcdhP2PExtendCtx::RecvDupMasked(
    std::vector<std::string>* dup_masked_items) {
  auto memory_store = std::make_shared<MemoryCipherStore>();

  RecvDualMaskedSelf(memory_store);

  *dup_masked_items = memory_store->self_results();
}

void EcdhP2PExtendCtx::SendDupMasked(
    const std::vector<std::string>& dual_masked_items) {
  SendImpl(dual_masked_items, true);
}

void EcdhP2PExtendCtx::SendItems(const std::vector<std::string>& items) {
  SendImpl(items, false);
}

void EcdhP2PExtendCtx::RecvItems(std::vector<std::string>* items) {
  size_t batch_count = 0;
  while (true) {
    std::vector<std::string> recv_batch_items;
    RecvBatch(&recv_batch_items, batch_count);
    for (auto& item : recv_batch_items) {
      items->emplace_back(std::move(item));
    }
    if (recv_batch_items.empty()) {
      SPDLOG_INFO("{} recv last batch finished, batch_count={}",
                  options_.link_ctx->Id(), batch_count);
      break;
    }
    batch_count++;
  }
}

void EcdhP2PExtendCtx::ForwardBatch(const std::vector<std::string>& batch_items,
                                    int32_t batch_idx) {
  SendBatch(batch_items, batch_idx);
}

void EcdhP2PExtendCtx::SendImpl(const std::vector<std::string>& items,
                                bool dup_masked) {
  size_t batch_count = 0;
  size_t send_count = 0;
  while (true) {
    size_t curr_step_item_num =
        std::min(options_.batch_size, items.size() - send_count);
    std::vector<absl::string_view> batch_items;
    for (size_t j = 0; j < curr_step_item_num; ++j) {
      batch_items.emplace_back(items[send_count + j]);
    }

    if (dup_masked) {
      SendDualMaskedBatch(batch_items, batch_count);
    } else {
      SendBatch(batch_items, batch_count);
    }

    if (curr_step_item_num == 0) {
      SPDLOG_INFO("SendImpl:{}--finished, batch_count={}",
                  options_.link_ctx->Id(), batch_count);
      break;
    }
    send_count += curr_step_item_num;
    ++batch_count;
  }
}

// shuffled ecdh 3pc psi
ShuffleEcdh3PcPsi::ShuffleEcdh3PcPsi(Options options)
    : options_(std::move(std::move(options))) {
  SPU_ENFORCE(options_.link_ctx->WorldSize() == 3);

  private_key_.resize(kKeySize);
  SPU_ENFORCE(RAND_bytes(private_key_.data(), kKeySize) == 1,
              "Cannot create random private key");
  ecc_cryptor_ = CreateEccCryptor(options_.curve_type);
  ecc_cryptor_->SetPrivateKey(absl::MakeSpan(private_key_));
}

void ShuffleEcdh3PcPsi::MaskMaster(const std::vector<std::string>& self_items,
                                   std::vector<std::string>* masked_items) {
  SPDLOG_INFO("MaskMaster:{} begin", options_.link_ctx->Rank());
  if (IsMaster()) {
    // c
    // - mask and send [c]-->a z^c
    // - recv b-->[c] z^c^a^b
    auto c_a_ctx =
        CreateP2PCtx("MaskMaster", options_.link_ctx->NextRank(),
                     options_.dual_mask_size, options_.link_ctx->Rank());
    auto c_b_ctx =
        CreateP2PCtx("MaskMaster", options_.link_ctx->PrevRank(),
                     options_.dual_mask_size, options_.link_ctx->Rank());

    auto mask_send_self =
        std::async([&] { return c_a_ctx->MaskSendSelf(self_items); });
    auto recv_triple_masked =
        std::async([&] { return c_b_ctx->RecvItems(masked_items); });

    mask_send_self.get();
    recv_triple_masked.get();

    SPDLOG_INFO("MaskMaster:{} recv masked master items:{}",
                options_.link_ctx->Rank(), masked_items->size());
  } else if (IsCalculator()) {
    // a
    // - c-->[a] recv z^c mask z^c^a then [a]-->b send z^c^a
    auto a_c_ctx =
        CreateP2PCtx("MaskMaster", options_.link_ctx->PrevRank(),
                     options_.dual_mask_size, options_.link_ctx->Rank());
    auto a_b_ctx =
        CreateP2PCtx("MaskMaster", options_.link_ctx->NextRank(),
                     options_.dual_mask_size, options_.link_ctx->NextRank());
    a_c_ctx->MaskPeerForward(a_b_ctx);
  } else /*IsPartner()*/ {
    // b
    // - a-->[b] recv z^c^a mask send [b]-->c z^c^a^b
    auto b_a_ctx =
        CreateP2PCtx("MaskMaster", options_.link_ctx->PrevRank(),
                     options_.dual_mask_size, options_.link_ctx->Rank());
    auto b_c_ctx =
        CreateP2PCtx("MaskMaster", options_.link_ctx->NextRank(),
                     options_.dual_mask_size, options_.link_ctx->NextRank());
    b_a_ctx->MaskPeerForward(b_c_ctx, options_.dual_mask_size);
  }
}

void ShuffleEcdh3PcPsi::PartnersPsi(const std::vector<std::string>& self_items,
                                    std::vector<std::string>* results) {
  if (IsPartner()) {
    std::vector<std::string> shuffle_inputs = self_items;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(shuffle_inputs.begin(), shuffle_inputs.end(), rng);

    PartnersPsiImpl(shuffle_inputs, results);
  } else {
    PartnersPsiImpl(self_items, results);
  }
}

void ShuffleEcdh3PcPsi::PartnersPsiImpl(
    const std::vector<std::string>& self_items,
    std::vector<std::string>* results) {
  SPDLOG_INFO("PartnersPsi:{} begin", options_.link_ctx->Rank());
  // a - b psi
  if (IsCalculator()) {
    std::vector<std::string> self_result;
    std::vector<std::string> peer_result;

    auto context =
        CreateP2PCtx("PartnersPsi", options_.link_ctx->NextRank(),
                     ecc_cryptor_->GetMaskLength(), options_.link_ctx->Rank());
    // check config
    context->CheckConfig();
    std::future<void> f_mask_self =
        std::async([&] { return context->MaskSendSelf(self_items); });
    std::future<void> f_mask_peer =
        std::async([&] { return context->MaskRecvPeer(&peer_result); });
    std::future<void> f_recv_peer =
        std::async([&] { return context->RecvDupMasked(&self_result); });

    f_mask_self.get();
    f_mask_peer.get();
    f_recv_peer.get();

    SPDLOG_INFO("PartnersPsi:{}--self_result:{}, peer_result:{}",
                options_.link_ctx->Rank(), self_result.size(),
                peer_result.size());

    // calc intersection_ab
    std::sort(self_result.begin(), self_result.end());
    std::sort(peer_result.begin(), peer_result.end());

    std::vector<std::string> intersection;
    std::set_intersection(self_result.begin(), self_result.end(),
                          peer_result.begin(), peer_result.end(),
                          std::back_inserter(intersection));
    // send to master
    auto a_c_ctx = CreateP2PCtx("PartnersPsi", options_.master_rank,
                                options_.dual_mask_size, options_.master_rank);
    a_c_ctx->SendItems(intersection);

    SPDLOG_INFO("PartnersPsi:{}--send to master_{}, intersection size:{}",
                options_.link_ctx->Rank(), options_.master_rank,
                intersection.size());

  } else if (IsPartner()) {
    auto context = CreateP2PCtx("PartnersPsi", options_.link_ctx->PrevRank(),
                                ecc_cryptor_->GetMaskLength(),
                                options_.link_ctx->PrevRank());
    // check config
    context->CheckConfig();
    std::future<void> f_mask_self =
        std::async([&] { return context->MaskSendSelf(self_items); });
    // shuffle x^a^b
    std::future<void> f_mask_peer =
        std::async([&] { return context->MaskShufflePeer(); });

    f_mask_self.get();
    f_mask_peer.get();
  } else {
    // c
    // recv result
    auto c_a_ctx =
        CreateP2PCtx("PartnersPsi", options_.link_ctx->NextRank(),
                     ecc_cryptor_->GetMaskLength(), options_.link_ctx->Rank());
    c_a_ctx->RecvItems(results);

    SPDLOG_INFO("PartnersPsi:{}--recv partner psi items:{}",
                options_.link_ctx->Rank(), results->size());
  }
}

void ShuffleEcdh3PcPsi::FinalPsi(
    const std::vector<std::string>& self_items,
    const std::vector<std::string>& master_items,
    const std::vector<std::string>& partners_result,
    std::vector<std::string>* results) {
  if (IsMaster()) {
    std::vector<std::string> masked_partners_items;
    for (const auto& masked : Mask(ecc_cryptor_, partners_result)) {
      masked_partners_items.emplace_back(masked.substr(
          masked.length() - options_.dual_mask_size, options_.dual_mask_size));
    }

    std::sort(masked_partners_items.begin(), masked_partners_items.end());

    for (uint32_t index = 0; index < master_items.size(); index++) {
      if (std::binary_search(masked_partners_items.begin(),
                             masked_partners_items.end(),
                             master_items[index])) {
        SPU_ENFORCE(index < self_items.size());
        results->push_back(self_items[index]);
      }
    }
  }
}

std::shared_ptr<EcdhP2PExtendCtx> ShuffleEcdh3PcPsi::CreateP2PCtx(
    const std::string& link_id_prefix, size_t dst_rank, size_t dual_mask_size,
    size_t target_rank) {
  EcdhPsiOptions opts;
  opts.link_ctx = CreateP2PLinkCtx(link_id_prefix, options_.link_ctx, dst_rank);
  opts.ecc_cryptor = ecc_cryptor_;
  opts.dual_mask_size = dual_mask_size;
  if (target_rank != yacl::link::kAllRank) {
    SPU_ENFORCE(target_rank == options_.link_ctx->Rank() ||
                target_rank == dst_rank);
    opts.target_rank = target_rank == dst_rank ? opts.link_ctx->NextRank()
                                               : opts.link_ctx->Rank();
  } else {
    opts.target_rank = target_rank;
  }

  return std::make_shared<EcdhP2PExtendCtx>(opts);
}

size_t ShuffleEcdh3PcPsi::GetPartnersPsiPeerRank() {
  SPU_ENFORCE(!IsMaster());
  if (IsCalculator()) {
    return options_.link_ctx->NextRank();
  } else {
    return options_.link_ctx->PrevRank();
  }
}

}  // namespace spu::psi