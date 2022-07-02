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

#include "spu/psi/core/ecdh_psi_mparty.h"

#include <future>
#include <utility>

#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "yasl/base/exception.h"

#include "spu/psi/cryptor/cryptor_selector.h"
#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

namespace spu::psi {

EcdhPsiMParty::EcdhPsiMParty(
    const std::shared_ptr<IBatchProvider>& batch_provider,
    const std::shared_ptr<ICipherStore>& cipher_store, CurveType curve_type,
    size_t batch_size) {
  private_key_.resize(kKeySize);
  YASL_ENFORCE(RAND_bytes(&private_key_[0], kKeySize) == 1,
               "Cannot create random private key");

  options_.curve_type = curve_type;
  options_.ecc_cryptor = CreateEccCryptor(curve_type);
  options_.ecc_cryptor->SetPrivateKey(absl::MakeSpan(private_key_));
  options_.batch_provider = batch_provider;
  options_.cipher_store = cipher_store;

  options_.target_rank = 0;
  options_.batch_size = batch_size;
}

EcdhPsiMParty::EcdhPsiMParty(const std::vector<std::string>& items,
                             const std::shared_ptr<ICipherStore>& cipher_store,
                             CurveType curve_type, size_t batch_size)
    : EcdhPsiMParty(std::make_shared<MemoryBatchProvider>(items), cipher_store,
                    curve_type, batch_size) {}

EcdhPsiMParty::~EcdhPsiMParty() { OPENSSL_cleanse(&private_key_[0], kKeySize); }

void EcdhPsiMParty::RunMaskRecvAndForward(
    const std::shared_ptr<yasl::link::Context>& link, size_t recv_rank,
    size_t send_rank, size_t dual_mask_size) {
  PsiOptions tmp_options = options_;

  tmp_options.link_ctx = link;

  EcdhPsiOp ecdh_psi_forward(tmp_options);

  // recv x^(...) from prev, send x^(...)^a to next

  return ecdh_psi_forward.MaskPeer(link->NextRank(), recv_rank, send_rank, link,
                                   dual_mask_size);
}

// recv from prev, mask and send to next
void EcdhPsiMParty::RunMaskSelfAndSend(
    const std::shared_ptr<yasl::link::Context>& link, size_t send_rank) {
  YASL_ENFORCE(link->Rank() != send_rank, "check send_rank({}) not self({})",
               send_rank, link->Rank());

  PsiOptions tmp_options = options_;

  tmp_options.link_ctx = link;

  EcdhPsiOp ecdh_psi(tmp_options);

  // send x^a to send_rank
  return ecdh_psi.MaskSelf(link->NextRank(), send_rank);
}

// recv from recv_rank, mask and store to cipher_batch,
// can do shuffle with stored data
void EcdhPsiMParty::RunMaskRecvAndStore(
    const std::shared_ptr<yasl::link::Context>& link, size_t recv_rank,
    size_t dual_mask_size) {
  PsiOptions tmp_options = options_;
  tmp_options.target_rank = link->Rank();

  tmp_options.link_ctx = link;

  EcdhPsiOp ecdh_psi(tmp_options);

  // recv x^{..} from recv_rank, mask and store
  return ecdh_psi.MaskPeer(link->Rank(), recv_rank, link->Rank(), link,
                           dual_mask_size);
}

// send data from tmp_batch_provider after intersection or shuffle
void EcdhPsiMParty::RunSendBatch(
    const std::shared_ptr<yasl::link::Context>& link, size_t send_rank,
    const std::shared_ptr<IBatchProvider>& shuffle_batch_provider) {
  PsiOptions tmp_options = options_;

  tmp_options.link_ctx = link;

  EcdhPsiOp ecdh_psi(tmp_options);

  // send shuffle_batch_provider to send_rank
  return ecdh_psi.SendBatch(send_rank, shuffle_batch_provider);
}

// recv from recv_rank, store to cipher_batch
void EcdhPsiMParty::RunRecvAndStore(
    const std::shared_ptr<yasl::link::Context>& link, size_t recv_rank,
    size_t dual_mask_size) {
  PsiOptions tmp_options = options_;

  tmp_options.link_ctx = link;

  EcdhPsiOp ecdh_psi(tmp_options);

  // recv x^{...} from recv_bank
  return ecdh_psi.RecvPeer(link->Rank(), recv_rank, link, dual_mask_size);
}

}  // namespace spu::psi
