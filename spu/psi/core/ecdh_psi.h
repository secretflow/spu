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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "yasl/link/link.h"

#include "spu/psi/cryptor/ecc_cryptor.h"
#include "spu/psi/provider/batch_provider.h"
#include "spu/psi/store/cipher_store.h"

#include "spu/psi/core/serializable.pb.h"

namespace spu::psi {

// I prefer 4096.
inline constexpr size_t kEcdhPsiBatchSize = 4096;

// Ecc256 requires 32 bytes.
inline constexpr size_t kKeySize = 32;
inline constexpr size_t kHashSize = kKeySize;

// The final comparison bytes.
// Hongcheng suggested that 90 bits would be enough. Here we give 96 bits.
inline constexpr size_t kFinalCompareBytes = 12;

namespace details {
struct EcdhBatch {
  // Pack all items in a single `std::string` to save bandwidth.
  //
  // If ecdh_round is kForSelf, each item size is `kHashSize`.
  // If ecdh_round is kForPeer, each item size is `kFinalCompareBytes`.
  std::string flatten_bytes;

  // Metadata.
  bool is_last_batch = false;

  yasl::Buffer Serialize() const {
    EcdhBatchProto proto;
    proto.set_flatten_bytes(flatten_bytes);
    proto.set_is_last_batch(is_last_batch);

    yasl::Buffer b(proto.ByteSizeLong());
    proto.SerializePartialToArray(b.data(), b.size());
    return b;
  }

  static EcdhBatch Deserialize(const yasl::Buffer& buf) {
    EcdhBatchProto proto;
    proto.ParseFromArray(buf.data(), buf.size());

    EcdhBatch batch;
    batch.flatten_bytes = proto.flatten_bytes();
    batch.is_last_batch = proto.is_last_batch();
    return batch;
  }
};
}  // namespace details

using FinishBatchHook = std::function<void(size_t)>;

struct PsiOptions {
  // Provides the link for the rank world.
  std::shared_ptr<yasl::link::Context> link_ctx;

  // Provides private inputs for ecdh-psi.
  std::shared_ptr<IEccCryptor> ecc_cryptor;

  // Provides private inputs for ecdh-psi.
  std::shared_ptr<IBatchProvider> batch_provider;

  // Cipher store
  std::shared_ptr<ICipherStore> cipher_store;

  // Points out which rank the psi results should be revealed.
  //
  // Allowed values:
  // - `link::kAllRank` i.e. std::numeric_limits<size_t>::max(), means reveal to
  // all rank
  // - otherwise the psi results should only revealed to the `target_rank`
  size_t target_rank = yasl::link::kAllRank;

  // These two options are used to control max memory consumption during the
  // PSI.
  //
  // Q: Why throttle control ?
  // A:
  // If we do not add the throttle controls, here is the risk:
  // - Alice has more powerful compute resources than bob, i.e. alice send
  // encrypted batches faster.
  // - Hence, bob might received lots of alice's batches. The memory will keep
  // increasing and finally we will get OOM. (Considering we need to support PSI
  // for billion items.)
  //
  // Q: What is `window_throttle_timeout_ms` & `window_size` ?
  // A: How much time alice should wait for bob's dual encrypted batches. If bob
  // have not finished dual encryption, alice will wait for Bob to catch up.
  // Alice will at most send `window_size` batches beforehand.
  size_t window_throttle_timeout_ms = 60 * 1000;
  size_t window_size = 8;

  // Fnish batch callback. Could be used for logging or update progress.
  FinishBatchHook on_batch_finished;

  // batch_size
  //     batch read from IBatchProvider
  //     batch compute dh mask
  //     batch send and read
  size_t batch_size = kEcdhPsiBatchSize;

  // curve_type
  CurveType curve_type = CurveType::Curve25519;
};

// RunEcdhPsi runs streaming-based in-memory ECDH-PSI.
void RunEcdhPsi(const PsiOptions& options);

// Simple wrapper for a common in memory psi case. It always use cpu based ecc
// cryptor.
std::vector<std::string> RunEcdhPsi(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const std::vector<std::string>& items, size_t target_rank,
    CurveType crve = CurveType::Curve25519);

// encapsulation of RunMaskSelf RunMaskPeer RunRecvPeer in ecdh_psi.cc
//  private_key_:   ec private key set by caller
//  target_rank:  decided by 3party ecdh psi protocol
//
class EcdhPsiOp {
 public:
  EcdhPsiOp(const PsiOptions& options);
  ~EcdhPsiOp() = default;

  // send_rank for link context world_size > 2
  void MaskSelf(size_t target_rank, size_t send_rank);

  /**
   * @brief recv masked_item, exec mask
   *
   * @param target_rank PSI result recv target
   * @param recv_rank   recv masked_data from recv_rank for world_size>2 link
   * @param send_rank   send dual mask to send_rank for world_size>2 link
   * @param link_ctx    link_ctx
   * @param dual_mask_size  dual_mask size, kFinalCompareBytes/kEccKeySize
   */
  void MaskPeer(size_t target_rank, size_t recv_rank, size_t send_rank,
                const std::shared_ptr<yasl::link::Context>& link_ctx,
                size_t dual_mask_size = kFinalCompareBytes);

  /**
   * @brief recv data
   *
   * @param target_rank  PSI result recv target
   * @param recv_rank    recv masked_data from recv_rank for world_size>2 link
   * @param link_ctx     link_ctx
   * @param dual_mask_size  mask size, kFinalCompareBytes/kEccKeySize
   */
  void RecvPeer(size_t target_rank, size_t recv_rank,
                const std::shared_ptr<yasl::link::Context>& link_ctx,
                size_t dual_mask_size = kFinalCompareBytes);

  /**
   * @brief  send data from tmp batchProvider
   *
   * @param send_rank  send data to send_rank for world_size>2 link
   * @param batch_provider  tmp batchProvider
   */
  void SendBatch(size_t send_rank,
                 const std::shared_ptr<IBatchProvider>& batch_provider);

 private:
  PsiOptions options_;
};

std::vector<std::string> HashInputs(const std::vector<std::string>& items);

}  // namespace spu::psi