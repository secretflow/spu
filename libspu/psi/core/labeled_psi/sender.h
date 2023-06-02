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

#include <memory>

#include "apsi/powers.h"
#include "apsi/psi_params.h"
#include "apsi/seal_object.h"
#include "seal/seal.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/link/link.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/sender_db.h"

namespace spu::psi {

class LabelPsiSender {
 public:
  explicit LabelPsiSender(std::shared_ptr<spu::psi::SenderDB> sender_db);

  /**
   * @brief  Receive PsiParams Request and Send PsiParams Response
   *
   * @param items_size
   * @param link_ctx
   */
  static void RunPsiParams(
      size_t items_size, const std::shared_ptr<yacl::link::Context>& link_ctx);

  /**
   * @brief Receive OPRF Request and Send OPRF Response
   *
   * @param oprf_server
   * @param link_ctx
   */
  static void RunOPRF(const std::shared_ptr<IEcdhOprfServer>& oprf_server,
                      const std::shared_ptr<yacl::link::Context>& link_ctx);

  /**
   * @brief Receive query_powers Request and Send polynomial ciphertext Response
   *
   * @param link_ctx
   */
  void RunQuery(const std::shared_ptr<yacl::link::Context>& link_ctx);

 private:
  std::shared_ptr<spu::psi::SenderDB> sender_db_;

  apsi::CryptoContext crypto_context_;
  seal::compr_mode_type compr_mode_ = seal::Serialization::compr_mode_default;

  apsi::PowersDag pd_;
};

}  // namespace spu::psi
