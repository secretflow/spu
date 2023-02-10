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

#include <memory>
#include <string>
#include <vector>

#include "libspu/psi/core/communication.h"
#include "libspu/psi/core/ecdh_psi.h"

namespace spu::psi {

class EcdhP2PExtendCtx : public EcdhPsiContext {
 public:
  explicit EcdhP2PExtendCtx(const EcdhPsiOptions& options);
  ~EcdhP2PExtendCtx() = default;

  // wrapper of `EcdhPsiContext::MaskSelf`
  void MaskSendSelf(const std::vector<std::string>& self_items);

  // wrapper of `EcdhPsiContext::MaskRecvPeer`
  void MaskRecvPeer(std::vector<std::string>* dup_masked_peer_items);

  // wrapper of `EcdhPsiContext::RecvDualMaskedSelf`
  void RecvDupMasked(std::vector<std::string>* dup_masked_items);

  // recv peer masked items, mask them again then use `forward_ctx` to forward
  void MaskPeerForward(const std::shared_ptr<EcdhP2PExtendCtx>& forward_ctx,
                       int32_t truncation_size = -1);

  // recv peer masked items, mask them again then shuffle and send them back
  void MaskShufflePeer();

  // send the duplicate masked items to peer
  void SendDupMasked(const std::vector<std::string>& dual_masked_items);

  void SendItems(const std::vector<std::string>& items);

  void RecvItems(std::vector<std::string>* items);

  // internal
  void ForwardBatch(const std::vector<std::string>& batch_items,
                    int32_t batch_idx);

 private:
  void SendImpl(const std::vector<std::string>& items, bool dup_masked);
};

//
// 3party ecdh psi algorithm.
//
//  master get the final intersection
//     calculator role is master-link->NextRank()
//     partner   role is master-link->PrevRank()
//
//  (calculator)       (partner)             (master)
//     alice           bob                    candy
// partners psi =============================================
//       |              | shuffle b items       |
//  x^a  |      x^a     |                       |
//       |   -------->  | x^a^b                 |
//       |              |                       |
//       |      y^b     | y^b                   |
//       |   <--------  |                       |
// y^b^a |              |                       |
//       |              | shuffle x^a^b         |
//       |      x^a^b   |                       |
//       |   <--------  |                       |
// calc intersection_ab |                       |
//       |            intersection_ab           |
//       |   ---------------------------------> | {intersection_ab}^c
// mask master ==============================================
//       |                 z^c                  |
//       |   <--------------------------------- |
// z^c^a |              |                       |
//       |    z^c^a     |                       |
//       |  -------->   |                       |
//       |              | calc {z^c^a}^b        |
//       |              |  send z^c^a^b         |
//       |              |  ------------------>  |
// calc result ==============================================
//       |              |                       |
//                                      calc intersection_abc
//
class ShuffleEcdh3PcPsi {
 public:
  struct Options {
    // Provides the link for the rank world.
    std::shared_ptr<yacl::link::Context> link_ctx;

    // master rank get final result
    size_t master_rank;

    // batch_size
    //     batch compute dh mask
    //     batch send and read
    size_t batch_size = kEcdhPsiBatchSize;

    size_t dual_mask_size = kFinalCompareBytes;

    // curve_type
    CurveType curve_type = CurveType::CURVE_25519;
  };

  explicit ShuffleEcdh3PcPsi(Options options);
  ~ShuffleEcdh3PcPsi() = default;

  // only master rank can get masked items
  void MaskMaster(const std::vector<std::string>& self_items,
                  std::vector<std::string>* masked_items);

  // master / calculator can get results
  void PartnersPsi(const std::vector<std::string>& self_items,
                   std::vector<std::string>* results);

  // only master rank can get results
  void FinalPsi(const std::vector<std::string>& self_items,
                const std::vector<std::string>& master_items,
                const std::vector<std::string>& partners_result,
                std::vector<std::string>* results);

 private:
  std::shared_ptr<EcdhP2PExtendCtx> CreateP2PCtx(
      const std::string& link_id_prefix, size_t dst_rank, size_t dual_mask_size,
      size_t target_rank);

  size_t GetPartnersPsiPeerRank();

  void PartnersPsiImpl(const std::vector<std::string>& self_items,
                       std::vector<std::string>* results);

  bool IsCalculator() {
    return options_.link_ctx->PrevRank() == options_.master_rank;
  }

  bool IsPartner() {
    return options_.link_ctx->NextRank() == options_.master_rank;
  }

  bool IsMaster() { return options_.link_ctx->Rank() == options_.master_rank; }

  Options options_;

  std::shared_ptr<IEccCryptor> ecc_cryptor_;
  // curve 25519 dh private key, 32B
  std::vector<uint8_t> private_key_;
};

}  // namespace spu::psi
