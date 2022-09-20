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

#include "spu/psi/operator/kkrt_2party_psi.h"

#include "yasl/crypto/hash_util.h"
#include "yasl/utils/parallel.h"

namespace spu::psi {

KkrtPsiOperator::KkrtPsiOperator(const Options& options)
    : PsiBaseOperator(options.link_ctx), options_(options) {}

std::vector<std::string> KkrtPsiOperator::OnRun(
    const std::vector<std::string>& inputs) {
  std::vector<std::string> res;

  // hash items to uint128_t
  std::vector<uint128_t> items_hash(inputs.size());
  yasl::parallel_for(0, inputs.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      items_hash[idx] = yasl::crypto::Blake3_128(inputs[idx]);
    }
  });

  if (options_.receiver_rank == link_ctx_->Rank()) {
    yasl::BaseSendOptions send_opts;

    GetKkrtOtReceiverOptions(options_.link_ctx, options_.num_ot, &send_opts);

    std::vector<size_t> kkrt_psi_result =
        KkrtPsiRecv(options_.link_ctx, send_opts, items_hash);

    for (auto index : kkrt_psi_result) {
      res.emplace_back(inputs[index]);
    }
  } else {
    yasl::BaseRecvOptions recv_opts;

    GetKkrtOtSenderOptions(options_.link_ctx, options_.num_ot, &recv_opts);

    KkrtPsiSend(options_.link_ctx, recv_opts, items_hash);
  }

  return res;
}

}  // namespace spu::psi
