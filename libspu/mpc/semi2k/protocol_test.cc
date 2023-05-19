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

#include "libspu/mpc/semi2k/protocol.h"

#include <mutex>

#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api_test.h"
#include "libspu/mpc/semi2k/beaver/ttp_server/beaver_server.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::SEMI2K);
  conf.set_field(field);
  return conf;
}

std::once_flag init_server;
std::unique_ptr<brpc::Server> server;
std::string server_host;

void InitBeaverServer() {
  std::call_once(init_server, []() {
    server = semi2k::beaver::ttp_server::RunServer(0);
    server_host = fmt::format("127.0.0.1:{}", server->listen_address().port);
  });
}

std::unique_ptr<SPUContext> makeTTPSemi2kProtocol(
    const RuntimeConfig& rt, const std::shared_ptr<yacl::link::Context>& lctx) {
  InitBeaverServer();
  RuntimeConfig ttp_rt = rt;

  ttp_rt.set_beaver_type(RuntimeConfig_BeaverType_TrustedThirdParty);
  auto* ttp = ttp_rt.mutable_ttp_beaver_config();
  ttp->set_adjust_rank(lctx->WorldSize() - 1);
  ttp->set_server_host(server_host);

  return makeSemi2kProtocol(ttp_rt, lctx);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ApiTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ArithmeticTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
      ;
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, BooleanTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
      ;
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ConversionTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
      ;
    });

}  // namespace spu::mpc::test
