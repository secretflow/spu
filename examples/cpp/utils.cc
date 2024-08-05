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

#include "examples/cpp/utils.h"

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"

#include "libspu/core/config.h"

#include "libspu/spu.pb.h"

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:39530,127.0.0.1:39531"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));
llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));
llvm::cl::opt<uint32_t> ProtocolKind(
    "protocol_kind", llvm::cl::init(2),
    llvm::cl::desc("1 for REF2k, 2 for SEMI2k, 3 for ABY3, 4 for Cheetah"));
llvm::cl::opt<uint32_t> Field(
    "field", llvm::cl::init(2),
    llvm::cl::desc("1 for Ring32, 2 for Ring64, 3 for Ring128"));
llvm::cl::opt<bool> EngineTrace("engine_trace", llvm::cl::init(false),
                                llvm::cl::desc("Enable trace info"));

std::shared_ptr<yacl::link::Context> MakeLink(const std::string& parties,
                                              size_t rank) {
  yacl::link::ContextDesc lctx_desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const auto id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }
  auto lctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

std::unique_ptr<spu::SPUContext> MakeSPUContext() {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());

  spu::RuntimeConfig config;
  config.set_protocol(static_cast<spu::ProtocolKind>(ProtocolKind.getValue()));
  config.set_field(static_cast<spu::FieldType>(Field.getValue()));

  populateRuntimeConfig(config);

  config.set_enable_action_trace(EngineTrace.getValue());
  config.set_enable_type_checker(EngineTrace.getValue());

  return std::make_unique<spu::SPUContext>(config, lctx);
}
