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
#include "libspu/mpc/cheetah/protocol.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(size_t field) {
  RuntimeConfig conf;
  ProtocolConfig proto_conf;
  CheetahConfig cheetah_conf;
  cheetah_conf.set_ot_kind(CheetahOtKind::YACL_Softspoken);
  proto_conf.set_kind(ProtocolKind::CHEETAH);
  proto_conf.set_field(field);
  *proto_conf.mutable_cheetah_config() = std::move(cheetah_conf);
  *conf.mutable_protocol() = std::move(proto_conf);
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ArithmeticTest,
    testing::Combine(testing::Values(makeCheetahProtocol),  //
                     testing::Values(makeConfig(32),        //
                                     makeConfig(64)),       //
                     testing::Values(2)),                   //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BooleanTest,
    testing::Combine(testing::Values(makeCheetahProtocol),  //
                     testing::Values(makeConfig(32),        //
                                     makeConfig(64),        //
                                     makeConfig(128)),      //
                     testing::Values(2)),                   //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ConversionTest,
    testing::Combine(testing::Values(makeCheetahProtocol),  //
                     testing::Values(makeConfig(32),        //
                                     makeConfig(64)),       //
                     testing::Values(2)),                   //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ConversionTest);

}  // namespace spu::mpc::test
