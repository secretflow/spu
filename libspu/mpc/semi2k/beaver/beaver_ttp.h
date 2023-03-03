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
#include <variant>

#include "brpc/channel.h"
#include "yacl/link/context.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/beaver_interface.h"

#include "libspu/mpc/semi2k/beaver/ttp_server/service.pb.h"

namespace spu::mpc::semi2k {

class BeaverTtp final : public Beaver {
 public:
  struct Options {
    std::string server_host;
    std::string session_id;
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

  PrgCounter counter_;

  Options options_;

  size_t child_counter_;

  mutable brpc::Channel channel_;

 public:
  explicit BeaverTtp(std::shared_ptr<yacl::link::Context> lctx, Options ops);

  ~BeaverTtp() override;

  Triple Mul(FieldType field, size_t size) override;

  Triple And(FieldType field, size_t size) override;

  Triple Dot(FieldType field, size_t M, size_t N, size_t K) override;

  Pair Trunc(FieldType field, size_t size, size_t bits) override;

  Triple TruncPr(FieldType field, size_t size, size_t bits) override;

  ArrayRef RandBit(FieldType field, size_t size) override;

  std::unique_ptr<Beaver> Spawn() override;
};

}  // namespace spu::mpc::semi2k
