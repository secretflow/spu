// Copyright 2023 Ant Group Co., Ltd.
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
#include "absl/types/span.h"
#include "yacl/base/dynamic_bitset.h"
#include "yacl/kernel/algorithms/ot_store.h"
#include "yacl/link/link.h"
namespace spu::mpc::spdz2k {

// TODO: maybe it can move to yacl
void KosOtExtSend(const std::shared_ptr<yacl::link::Context>& ctx,
                  const std::shared_ptr<yacl::crypto::OtRecvStore>& base_ot,
                  absl::Span<uint128_t> send_blocks, uint128_t& delta);

void KosOtExtRecv(const std::shared_ptr<yacl::link::Context>& ctx,
                  const std::shared_ptr<yacl::crypto::OtSendStore>& base_ot,
                  const std::vector<bool>& choices,
                  absl::Span<uint128_t> recv_blocks);

};  // namespace spu::mpc::spdz2k