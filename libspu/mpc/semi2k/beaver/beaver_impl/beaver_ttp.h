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

#include "brpc/channel.h"
#include "yacl/base/buffer.h"
#include "yacl/link/context.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/beaver_interface.h"

#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/service.pb.h"

namespace spu::mpc::semi2k {

class BeaverTtp final : public Beaver {
 public:
  struct Options {
    std::string server_host;
    // asym_crypto_schema: support ["SM2"]
    // Will support 25519 in the future, after yacl supported it.
    std::string asym_crypto_schema;
    // TODO: Remote Attestation
    yacl::Buffer server_public_key;
    size_t adjust_rank;

    std::string brpc_channel_protocol = "baidu_std";
    std::string brpc_channel_connection_type = "single";
    std::string brpc_load_balancer_name;
    int32_t brpc_timeout_ms = 10 * 1000;
    int32_t brpc_max_retry = 5;

    // TODO: TLS ops for client/server two-way authentication
  };

 private:
  std::shared_ptr<yacl::link::Context> lctx_;

  PrgSeed seed_;

  std::vector<PrgSeedBuff> encrypted_seeds_;

  PrgCounter counter_;

  Options options_;

  mutable brpc::Channel channel_;

 public:
  explicit BeaverTtp(std::shared_ptr<yacl::link::Context> lctx, Options ops);

  ~BeaverTtp() override = default;

  Triple Mul(FieldType field, int64_t size, ReplayDesc* x_desc = nullptr,
             ReplayDesc* y_desc = nullptr,
             ElementType eltype = ElementType::kRing) override;

  Pair MulPriv(FieldType field, int64_t size,
               ElementType eltype = ElementType::kRing) override;

  Pair Square(FieldType field, int64_t size,
              ReplayDesc* x_desc = nullptr) override;

  Triple And(int64_t size) override;

  Triple Dot(FieldType field, int64_t m, int64_t n, int64_t k,
             ReplayDesc* x_desc = nullptr,
             ReplayDesc* y_desc = nullptr) override;

  Pair Trunc(FieldType field, int64_t size, size_t bits) override;

  Triple TruncPr(FieldType field, int64_t size, size_t bits) override;

  Array RandBit(FieldType field, int64_t size) override;

  Pair PermPair(FieldType field, int64_t size, size_t perm_rank,
                absl::Span<const int64_t> perm_vec) override;

  std::unique_ptr<Beaver> Spawn() override;

  Pair Eqz(FieldType field, int64_t size) override;
};

}  // namespace spu::mpc::semi2k
