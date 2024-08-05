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

#include "libspu/mpc/api_test.h"
#include "libspu/mpc/cheetah/protocol.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig() {
  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::CHEETAH);
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ApiTest,
    testing::Combine(testing::Values(makeCheetahProtocol),  //
                     testing::Values(makeConfig()),         //
                     testing::Values(2),                    //
                     testing::Values(FM32, FM64, FM128)),   //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<2>(p.param), std::get<3>(p.param));
    });

}  // namespace spu::mpc::test
