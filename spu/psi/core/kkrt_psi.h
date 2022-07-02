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
#include <unordered_map>
#include <vector>

#include "yasl/link/link.h"
#include "yasl/mpctools/ot/options.h"

//
// implementation of KKRT16 PSI protocol
// https://eprint.iacr.org/2016/799.pdf
//
// use Stash less Cuckoo hash optimization
// Reference:
// PSZ18 Scalable private set intersection based on ot extension
// https://eprint.iacr.org/2016/930.pdf
//

namespace spu::psi {

struct KkrtPsiOptions {
  // batch size the receiver send corrections
  size_t ot_batch_size = 128;

  // batch size the sender used to send oprf encode
  size_t psi_batch_size = 128;

  // cuckoo hash parameter
  // now use stashless setting
  // stash_size = 0  cuckoo_hash_num =3
  // use stat_sec_param = 40
  size_t cuckoo_hash_num = 3;
  size_t stash_size = 0;
  size_t stat_sec_param = 40;
};

void GetKkrtOtSenderOptions(
    const std::shared_ptr<yasl::link::Context>& link_ctx, const size_t num_ot,
    yasl::BaseRecvOptions* recv_opts);

void GetKkrtOtReceiverOptions(
    const std::shared_ptr<yasl::link::Context>& link_ctx, const size_t num_ot,
    yasl::BaseSendOptions* send_opts);

KkrtPsiOptions GetDefaultKkrtPsiOptions();

//
// sender and receiver psi input data shoud be prepocessed using hash algorithm.
// like sha256 or blake2/blake3 hash algorithm or aes_ecb(key, x)^x
//
void KkrtPsiSend(const std::shared_ptr<yasl::link::Context>& link_ctx,
                 const KkrtPsiOptions& kkrt_psi_options,
                 const yasl::BaseRecvOptions& base_options,
                 const std::vector<uint128_t>& items_hash);

std::vector<std::size_t> KkrtPsiRecv(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const KkrtPsiOptions& kkrt_psi_options,
    const yasl::BaseSendOptions& base_options,
    const std::vector<uint128_t>& items_hash);

void KkrtPsiSend(const std::shared_ptr<yasl::link::Context>& link_ctx,
                 const yasl::BaseRecvOptions& base_options,
                 const std::vector<uint128_t>& items_hash);

std::vector<std::size_t> KkrtPsiRecv(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const yasl::BaseSendOptions& base_options,
    const std::vector<uint128_t>& items_hash);

}  // namespace spu::psi
