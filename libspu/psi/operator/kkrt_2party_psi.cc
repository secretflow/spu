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

#include "libspu/psi/operator/kkrt_2party_psi.h"

#include <memory>

#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/utils/parallel.h"

#include "libspu/psi/operator/factory.h"

namespace spu::psi {

KkrtPsiOperator::Options KkrtPsiOperator::ParseConfig(
    const MemoryPsiConfig& config,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  return {lctx, config.receiver_rank()};
}

std::vector<std::string> KkrtPsiOperator::OnRun(
    const std::vector<std::string>& inputs) {
  std::vector<std::string> res;

  // hash items to uint128_t
  std::vector<uint128_t> items_hash(inputs.size());
  yacl::parallel_for(0, inputs.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      items_hash[idx] = yacl::crypto::Blake3_128(inputs[idx]);
    }
  });

  if (options_.receiver_rank == link_ctx_->Rank()) {
    auto ot_send = GetKkrtOtReceiverOptions(options_.link_ctx, options_.num_ot);
    std::vector<size_t> kkrt_psi_result =
        KkrtPsiRecv(options_.link_ctx, ot_send, items_hash);

    for (auto index : kkrt_psi_result) {
      res.emplace_back(inputs[index]);
    }
  } else {
    auto ot_recv = GetKkrtOtSenderOptions(options_.link_ctx, options_.num_ot);
    KkrtPsiSend(options_.link_ctx, ot_recv, items_hash);
  }

  return res;
}

namespace {

std::unique_ptr<PsiBaseOperator> CreateOperator(
    const MemoryPsiConfig& config,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto options = KkrtPsiOperator::ParseConfig(config, lctx);
  return std::make_unique<KkrtPsiOperator>(options);
}

REGISTER_OPERATOR(KKRT_PSI_2PC, CreateOperator);

}  // namespace

}  // namespace spu::psi
