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

#include "spu/psi/operator/ecdh_3party_psi.h"

#include <future>
#include <random>

#include "fmt/format.h"
#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/link/context.h"
#include "yasl/link/link.h"
#include "yasl/utils/serialize.h"

#include "spu/psi/cryptor/cryptor_selector.h"

namespace spu::psi {

namespace {
constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;

}  // namespace

Ecdh3PartyPsiOperator::Ecdh3PartyPsiOperator(const Options& options)
    : PsiBaseOperator(options.link_ctx), options_(options), handler_(nullptr) {
  options_.link_ctx->SetRecvTimeout(kLinkRecvTimeout);

  ShuffleEcdh3PcPsi::Options opts;
  opts.link_ctx = options_.link_ctx;
  opts.master_rank = options_.master_rank;
  opts.batch_size = options_.batch_size;
  opts.dual_mask_size = options_.dual_mask_size;
  opts.curve_type = options_.curve_type;

  handler_ = std::make_shared<ShuffleEcdh3PcPsi>(opts);
}

std::vector<std::string> Ecdh3PartyPsiOperator::OnRun(
    const std::vector<std::string>& inputs) {
  std::vector<std::string> results;
  std::vector<std::string> masked_master_items;
  std::vector<std::string> partner_psi_items;

  auto mask_master = std::async(
      [&] { return handler_->MaskMaster(inputs, &masked_master_items); });
  auto partner_psi = std::async(
      [&] { return handler_->PartnersPsi(inputs, &partner_psi_items); });

  mask_master.get();
  partner_psi.get();

  handler_->FinalPsi(inputs, masked_master_items, partner_psi_items, &results);

  return results;
}

}  // namespace spu::psi
