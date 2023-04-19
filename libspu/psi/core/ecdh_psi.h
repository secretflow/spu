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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "yacl/link/link.h"

#include "libspu/psi/core/communication.h"
#include "libspu/psi/cryptor/ecc_cryptor.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/cipher_store.h"

#include "libspu/psi/utils/serializable.pb.h"

namespace spu::psi {

using FinishBatchHook = std::function<void(size_t)>;

struct EcdhPsiStatistics {
  size_t self_item_count = 0;
  size_t peer_item_count = 0;
};

struct EcdhPsiOptions {
  // Provides the link for the rank world.
  std::shared_ptr<yacl::link::Context> link_ctx;

  // ic_mode = true: 互联互通模式，对方可以是非隐语应用
  // Interconnection mode, the other side can be non-secretflow application
  bool ic_mode = false;

  // Provides private inputs for ecdh-psi.
  std::shared_ptr<IEccCryptor> ecc_cryptor;

  size_t dual_mask_size = kFinalCompareBytes;

  // batch_size
  //     batch compute dh mask
  //     batch send and read
  size_t batch_size = kEcdhPsiBatchSize;

  // Points out which rank the psi results should be revealed.
  //
  // Allowed values:
  // - `link::kAllRank` i.e. std::numeric_limits<size_t>::max(), means reveal
  // to all rank
  // - otherwise the psi results should only revealed to the `target_rank`
  size_t target_rank = yacl::link::kAllRank;

  // Finish batch callback. Could be used for logging or update progress.
  FinishBatchHook on_batch_finished;

  // Collect information such as the input size of psi.
  EcdhPsiStatistics* statistics = nullptr;
};

// batch handler for 2-party ecdh psi
class EcdhPsiContext {
 public:
  explicit EcdhPsiContext(EcdhPsiOptions options);
  ~EcdhPsiContext() = default;

  void CheckConfig();

  // one of main steps for 2 party ecdh psi
  // mask self items then send them to peer
  void MaskSelf(const std::shared_ptr<IBatchProvider>& batch_provider);

  // one of main steps for 2 party ecdh psi
  // recv peer's masked items, then masked them again and send back
  // cipher_store: available only when target_rank equal to self_rank
  void MaskPeer(const std::shared_ptr<ICipherStore>& cipher_store);

  // one of main steps for 2 party ecdh psi
  // recv dual masked self items from peer
  // cipher_store: available only when target_rank equal to self_rank
  void RecvDualMaskedSelf(const std::shared_ptr<ICipherStore>& cipher_store);

  size_t PeerRank() { return options_.link_ctx->NextRank(); }

  std::string Id() { return options_.link_ctx->Id(); }

 protected:
  void SendBatch(const std::vector<std::string>& batch_items, int32_t batch_idx,
                 std::string_view tag = "");

  void SendBatch(const std::vector<std::string_view>& batch_items,
                 int32_t batch_idx, std::string_view tag = "");

  void RecvBatch(std::vector<std::string>* items, int32_t batch_idx,
                 std::string_view tag = "");

  void SendDualMaskedBatch(const std::vector<std::string>& batch_items,
                           int32_t batch_idx, std::string_view tag = "");

  void SendDualMaskedBatch(const std::vector<std::string_view>& batch_items,
                           int32_t batch_idx, std::string_view tag = "");

  void RecvDualMaskedBatch(std::vector<std::string>* items, int32_t batch_idx,
                           std::string_view tag = "");

 private:
  bool CanTouchResults() const {
    return options_.target_rank == yacl::link::kAllRank ||
           options_.target_rank == options_.link_ctx->Rank();
  }

  bool PeerCanTouchResults() const {
    return options_.target_rank == yacl::link::kAllRank ||
           options_.target_rank == options_.link_ctx->NextRank();
  }

 protected:
  EcdhPsiOptions options_;

  std::shared_ptr<yacl::link::Context> main_link_ctx_;
  std::shared_ptr<yacl::link::Context> dual_mask_link_ctx_;
};

void RunEcdhPsi(const EcdhPsiOptions& options,
                const std::shared_ptr<IBatchProvider>& batch_provider,
                const std::shared_ptr<ICipherStore>& cipher_store);

// Simple wrapper for a common in memory psi case. It always use cpu based
// ecc cryptor.
std::vector<std::string> RunEcdhPsi(
    const std::shared_ptr<yacl::link::Context>& link_ctx,
    const std::vector<std::string>& items, size_t target_rank,
    CurveType curve = CurveType::CURVE_25519,
    size_t batch_size = kEcdhPsiBatchSize);

}  // namespace spu::psi
