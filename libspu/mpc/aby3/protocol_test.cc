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

#include "libspu/mpc/aby3/protocol.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/api_test.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(size_t field) {
  RuntimeConfig conf;
  ProtocolConfig proto_conf;
  proto_conf.set_kind(ProtocolKind::ABY3);
  proto_conf.set_field(field);
  *conf.mutable_protocol() = std::move(proto_conf);
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Aby3, ApiTest,
    testing::Combine(testing::Values(makeAby3Protocol),  //
                     testing::Values(makeConfig(32),     //
                                     makeConfig(64),     //
                                     makeConfig(128)),   //
                     testing::Values(3)),                //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Aby3, ArithmeticTest,
    testing::Combine(testing::Values(makeAby3Protocol),  //
                     testing::Values(makeConfig(32),     //
                                     makeConfig(64),     //
                                     makeConfig(128)),   //
                     testing::Values(3)),                //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Aby3, BooleanTest,
    testing::Combine(testing::Values(makeAby3Protocol),  //
                     testing::Values(makeConfig(32),     //
                                     makeConfig(64),     //
                                     makeConfig(128)),   //
                     testing::Values(3)),                //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Aby3, ConversionTest,
    testing::Combine(testing::Values(makeAby3Protocol),  //
                     testing::Values(makeConfig(32),     //
                                     makeConfig(64),     //
                                     makeConfig(128)),   //
                     testing::Values(3)),                //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

}  // namespace spu::mpc::test
