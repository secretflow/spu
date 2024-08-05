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

#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/spdz2k/protocol.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig() {
  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::SEMI2K);  // FIXME:
  return conf;
}

std::unique_ptr<SPUContext> makeMpcSpdz2kProtocol(
    const RuntimeConfig& rt, const std::shared_ptr<yacl::link::Context>& lctx) {
  RuntimeConfig mpc_rt = rt;
  mpc_rt.set_beaver_type(RuntimeConfig_BeaverType_MultiParty);

  return makeSpdz2kProtocol(mpc_rt, lctx);
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Spdz2k, ArithmeticTest,
    testing::Values(std::tuple{CreateObjectFn(makeSpdz2kProtocol, "tfp"),
                               makeConfig(), 2, FM64},
                    std::tuple{CreateObjectFn(makeMpcSpdz2kProtocol, "mpc"),
                               makeConfig(), 2, FM32}),
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<2>(p.param), std::get<3>(p.param));
    });

// TODO : improve performance of boolean share and conversion in offline phase
INSTANTIATE_TEST_SUITE_P(
    Spdz2k, BooleanTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSpdz2kProtocol,
                                                    "tfp")),  //
                     testing::Values(makeConfig()),           //
                     testing::Values(2),                      //
                     testing::Values(FM64)),                  //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<2>(p.param), std::get<3>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Spdz2k, ConversionTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSpdz2kProtocol,
                                                    "tfp")),  //
                     testing::Values(makeConfig()),           //
                     testing::Values(2),                      //
                     testing::Values(FM64)),                  //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<2>(p.param), std::get<3>(p.param));
    });

}  // namespace spu::mpc::test
