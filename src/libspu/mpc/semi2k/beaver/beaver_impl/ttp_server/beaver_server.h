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

#include <memory>
#include <optional>

#include "brpc/server.h"
#include "yacl/base/buffer.h"

namespace spu::mpc::semi2k::beaver::ttp_server {

struct ServerOptions {
  int32_t port;
  // asym_crypto_schema: support ["SM2"]
  // Will support 25519 in the future, after yacl supported it.
  std::string asym_crypto_schema;
  yacl::Buffer server_private_key;
  std::optional<brpc::ServerSSLOptions> brpc_ssl_options;
};

std::unique_ptr<brpc::Server> RunServer(const ServerOptions& options);
int RunUntilAskedToQuit(const ServerOptions& options);

}  // namespace spu::mpc::semi2k::beaver::ttp_server
