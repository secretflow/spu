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
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "yacl/base/byte_container_view.h"
#include "yacl/link/link.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
// #include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/core/ecdh_oprf_psi.h"
#pragma once

#include "yacl/crypto/tools/prg.h"

#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/cipher_store.h"
#include "libspu/psi/utils/ub_psi_cache.h"

namespace spu::pir {
class LabeledEcdhOprfPsiServer {
 public:
  explicit LabeledEcdhOprfPsiServer(const spu::psi::EcdhOprfPsiOptions& options)
      : options_(options),
        oprf_server_id_(
            CreateEcdhOprfServer(options.oprf_type, options.curve_type)),
        oprf_server_label_(
            CreateEcdhOprfServer(options.oprf_type, options.curve_type)) {
    label_length_ = 0;
  }

  LabeledEcdhOprfPsiServer(const spu::psi::EcdhOprfPsiOptions& options,
                           yacl::ByteContainerView id_private_key,
                           yacl::ByteContainerView label_private_key,
                           size_t label_length)
      : options_(options),
        oprf_server_id_(CreateEcdhOprfServer(id_private_key, options.oprf_type,
                                             options.curve_type)),
        oprf_server_label_(CreateEcdhOprfServer(
            label_private_key, options.oprf_type, options.curve_type)),
        label_length_(label_length) {}

  size_t FullEvaluateAndSend(
      const std::shared_ptr<spu::psi::IBatchProvider>& batch_provider);

  void RecvBlindAndSendEvaluate();

  std::array<uint8_t, spu::psi::kEccKeySize> GetIDPrivateKey() {
    return oprf_server_id_->GetPrivateKey();
  }

  std::array<uint8_t, spu::psi::kEccKeySize> GetLabelPrivateKey() {
    return oprf_server_label_->GetPrivateKey();
  }

  size_t GetIDCompareLength() { return oprf_server_id_->GetCompareLength(); }

  void SetLabelLength(size_t label_byte_count) {
    label_length_ = label_byte_count;
  }

 private:
  spu::psi::EcdhOprfPsiOptions options_;

  std::shared_ptr<spu::psi::IEcdhOprfServer> oprf_server_id_;
  std::shared_ptr<spu::psi::IEcdhOprfServer> oprf_server_label_;

  size_t label_length_;
};

class LabeledEcdhOprfPsiClient {
 public:
  explicit LabeledEcdhOprfPsiClient(const spu::psi::EcdhOprfPsiOptions& options)
      : options_(options) {
    std::shared_ptr<spu::psi::IEcdhOprfClient> oprf_client =
        spu::psi::CreateEcdhOprfClient(options.oprf_type, options.curve_type);
    compare_length_ = oprf_client->GetCompareLength();
    ec_point_length_ = oprf_client->GetEcPointLength();
    label_length_ = 0;
  }

  explicit LabeledEcdhOprfPsiClient(const spu::psi::EcdhOprfPsiOptions& options,
                                    yacl::ByteContainerView private_key)
      : options_(options) {
    oprf_client_ = spu::psi::CreateEcdhOprfClient(private_key, options.oprf_type,
                                        options.curve_type);
    compare_length_ = oprf_client_->GetCompareLength();
    ec_point_length_ = oprf_client_->GetEcPointLength();
    label_length_ = 0;
  }

  void SetLabelLength(size_t label_byte_count) {
    label_length_ = label_byte_count;
  }
  void RecvFinalEvaluatedItems(std::vector<std::string>* server_ids,
                               std::vector<std::string>* server_labels);

  size_t SendBlindedItems(
      const std::unique_ptr<spu::psi::CsvBatchProvider>& batch_provider,
      std::vector<std::string>* client_ids);

  void RecvEvaluatedItems(std::vector<std::string>* client_ids,
                          std::vector<std::string>* client_labels);

  std::pair<std::vector<uint64_t>, std::vector<std::string>>
  FinalizeAndDecryptLabels(
      const std::shared_ptr<spu::psi::MemoryBatchProvider>& server_batch_provider,
      const std::vector<std::string>& client_ids,
      const std::vector<std::string>& client_labels);

 private:
  spu::psi::EcdhOprfPsiOptions options_;

  std::mutex mutex_;
  std::condition_variable queue_push_cv_;
  std::condition_variable queue_pop_cv_;
  std::queue<std::vector<std::shared_ptr<spu::psi::IEcdhOprfClient>>> oprf_client_queue_;
  std::shared_ptr<spu::psi::IEcdhOprfClient> oprf_client_ = nullptr;

  size_t compare_length_;
  size_t ec_point_length_;
  size_t label_length_;
};

}