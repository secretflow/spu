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

#include "spu/psi/core/ecdh_psi_3party.h"

#include <functional>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "fmt/format.h"
#include "yasl/base/exception.h"
#include "yasl/link/context.h"
#include "yasl/link/link.h"
#include "yasl/utils/serialize.h"

#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

namespace spu::psi {

namespace {
constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;
}

ShuffleEcdhPSI3Party::ShuffleEcdhPSI3Party(
    const std::shared_ptr<yasl::link::Context>& link, size_t master_rank,
    const std::shared_ptr<IBatchProvider>& batch_provider,
    const std::shared_ptr<ICipherStore>& cipher_store, CurveType curve_type,
    size_t batch_size)
    : link_(link),
      ecdh_psi_(batch_provider, cipher_store, curve_type, batch_size),
      master_rank_(master_rank) {
  link_->SetRecvTimeout(kLinkRecvTimeout);
}

ShuffleEcdhPSI3Party::ShuffleEcdhPSI3Party(
    const std::shared_ptr<yasl::link::Context>& link, size_t master_rank,
    const std::vector<std::string>& items,
    const std::shared_ptr<ICipherStore>& cipher_store, CurveType curve_type,
    size_t batch_size)
    : link_(link),
      ecdh_psi_(items, cipher_store, curve_type, batch_size),
      master_rank_(master_rank) {
  link_->SetRecvTimeout(kLinkRecvTimeout);
}

void ShuffleEcdhPSI3Party::RunEcdhPsiStep1() {
  std::shared_ptr<yasl::link::Context> link2 = link_->Spawn();
  if (link_->Rank() == master_rank_) {
    //[c]-->a-->b
    std::future<void> f_master_mask_self_send = std::async(
        [&] { return ecdh_psi_.RunMaskSelfAndSend(link_, link_->NextRank()); });

    // b-->[c]
    std::future<void> f_master_mask_recv_store = std::async([&] {
      return ecdh_psi_.RunRecvAndStore(link_, link_->PrevRank(),
                                       kFinalCompareBytes);
    });

    f_master_mask_self_send.get();
    f_master_mask_recv_store.get();

  } else if (link_->PrevRank() == master_rank_) {
    // [a]-->b send x^a
    std::future<void> f_master_prev_mask_self_send = std::async(
        [&] { return ecdh_psi_.RunMaskSelfAndSend(link2, link_->NextRank()); });

    // c-->[a]-->b recv z^c send z^c^a
    std::future<void> f_master_prev_mask_recv_forward = std::async([&] {
      return ecdh_psi_.RunMaskRecvAndForward(link_, link_->PrevRank(),
                                             link_->NextRank(), kHashSize);
    });

    // b-->[a] recv y^b
    std::future<void> f_master_prev_mask_recv_store = std::async([&] {
      return ecdh_psi_.RunMaskRecvAndStore(link_, link_->NextRank(), kHashSize);
    });

    f_master_prev_mask_self_send.get();
    f_master_prev_mask_recv_forward.get();
    f_master_prev_mask_recv_store.get();

  } else if (link_->NextRank() == master_rank_) {
    // [b]->a
    std::future<void> f_master_next_mask_self_send = std::async(
        [&] { return ecdh_psi_.RunMaskSelfAndSend(link_, link_->PrevRank()); });

    // a-->[b]
    std::future<void> f_master_next_mask_recv_store = std::async([&] {
      return ecdh_psi_.RunMaskRecvAndStore(link2, link_->PrevRank(), kHashSize);
    });

    // c-->a-->[b]-->c
    std::future<void> f_master_next_mask_recv_forward_toc = std::async([&] {
      return ecdh_psi_.RunMaskRecvAndForward(
          link_, link_->PrevRank(), link_->NextRank(), kFinalCompareBytes);
    });

    f_master_next_mask_self_send.get();
    f_master_next_mask_recv_store.get();
    f_master_next_mask_recv_forward_toc.get();
  }
}

void ShuffleEcdhPSI3Party::RunEcdhPsiStep2(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
  std::string step2_finish_ack_str = "step2 finished";

  if (link_->Rank() == master_rank_) {
    const auto tag = fmt::format("ShuffleEcdhPSI3Party:Step2:Recv:{}->{}",
                                 link_->NextRank(), link_->Rank());
    auto step2_ack_buf = link_->Recv(link_->NextRank(), tag);
    std::string step2_ack_str(step2_ack_buf.data<char>(), step2_ack_buf.size());

    YASL_ENFORCE_EQ(step2_ack_str, step2_finish_ack_str);
  } else if (link_->PrevRank() == master_rank_) {
    std::future<void> f_master_next_step2 = std::async([&] {
      return ecdh_psi_.RunRecvAndStore(link_, link_->NextRank(), kHashSize);
    });
    f_master_next_step2.get();

    link_->SendAsync(master_rank_,
                     yasl::ByteContainerView(step2_finish_ack_str),
                     fmt::format("ShuffleEcdhPSI3Party:Step2:Send:{}->{}",
                                 link_->Rank(), master_rank_));
  } else if (link_->NextRank() == master_rank_) {
    std::future<void> f_master_prev_step2 = std::async([&] {
      return ecdh_psi_.RunSendBatch(link_, link_->PrevRank(), batch_provider);
    });
    f_master_prev_step2.get();
  }
}

void ShuffleEcdhPSI3Party::RunEcdhPsiStep3(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
  if (link_->Rank() == master_rank_) {
    std::future<void> f_master_step2 = std::async([&] {
      return ecdh_psi_.RunMaskRecvAndStore(link_, link_->NextRank(),
                                           kFinalCompareBytes);
    });
    f_master_step2.get();
  } else if (link_->PrevRank() == master_rank_) {
    std::future<void> f_master_prev_step2 = std::async([&] {
      return ecdh_psi_.RunSendBatch(link_, master_rank_, batch_provider);
    });
    f_master_prev_step2.get();
  } else if (link_->NextRank() == master_rank_) {
    return;
  }
}

std::vector<std::string> RunShuffleEcdhPsi3Party(
    const std::shared_ptr<yasl::link::Context>& link, size_t master_rank,
    std::vector<std::string>& items, CurveType curve_type, size_t batch_size) {
  // shuffle items
  if (link->NextRank() == master_rank) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(items.begin(), items.end(), rng);
  }

  auto memory_store = std::make_shared<MemoryCipherStore>();
  ShuffleEcdhPSI3Party ecdh_3party_psi(link, master_rank, items, memory_store,
                                       curve_type, batch_size);

  std::future<void> f_dhpsi_3party_step1 =
      std::async([&] { return ecdh_3party_psi.RunEcdhPsiStep1(); });

  f_dhpsi_3party_step1.get();

  if (link->NextRank() == master_rank) {
    std::vector<std::string>& peer_result = memory_store->peer_results();
    std::sort(peer_result.begin(), peer_result.end());
    std::shared_ptr<IBatchProvider> shuffle_batch_provider =
        std::make_shared<MemoryBatchProvider>(peer_result);

    std::future<void> f_dhpsi_3party_step2 = std::async([&] {
      return ecdh_3party_psi.RunEcdhPsiStep2(shuffle_batch_provider);
    });
    f_dhpsi_3party_step2.get();
  } else {
    std::future<void> f_dhpsi_3party_step2 =
        std::async([&] { return ecdh_3party_psi.RunEcdhPsiStep2(); });
    f_dhpsi_3party_step2.get();
  }

  if (link->PrevRank() == master_rank) {
    std::vector<std::string>& self_result_master_next =
        memory_store->self_results();
    std::vector<std::string>& peer_result_master_next =
        memory_store->peer_results();

    std::sort(self_result_master_next.begin(), self_result_master_next.end());
    std::sort(peer_result_master_next.begin(), peer_result_master_next.end());

    std::vector<std::string> intersection_ab;
    std::set_intersection(
        self_result_master_next.begin(), self_result_master_next.end(),
        peer_result_master_next.begin(), peer_result_master_next.end(),
        std::back_inserter(intersection_ab));

    //
    std::shared_ptr<IBatchProvider> ab_batch_provider =
        std::make_shared<MemoryBatchProvider>(intersection_ab);

    std::future<void> f_dhpsi_3party_step3 = std::async(
        [&] { return ecdh_3party_psi.RunEcdhPsiStep3(ab_batch_provider); });
    f_dhpsi_3party_step3.get();
  } else {
    std::future<void> f_dhpsi_3party_step3 =
        std::async([&] { return ecdh_3party_psi.RunEcdhPsiStep3(); });
    f_dhpsi_3party_step3.get();
  }

  std::vector<std::string> ret;
  if (link->Rank() == master_rank) {
    std::vector<std::string>& self_result_master = memory_store->self_results();
    std::vector<std::string>& peer_result_master = memory_store->peer_results();

    std::sort(peer_result_master.begin(), peer_result_master.end());

    for (uint32_t index = 0; index < self_result_master.size(); index++) {
      if (std::binary_search(peer_result_master.begin(),
                             peer_result_master.end(),
                             self_result_master[index])) {
        YASL_ENFORCE(index < items.size());
        ret.push_back(items[index]);
      }
    }
  }

  return ret;
}

}  // namespace spu::psi
