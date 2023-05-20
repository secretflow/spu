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

#include "libspu/mpc/spdz2k/protocol.h"

#include "libspu/mpc/spdz2k/abprotocol_spdz2k_test.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::SEMI2K);  // FIXME:
  conf.set_field(field);
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Spdz2kArithmeticTest, ArithmeticTest,
    testing::Combine(testing::Values(makeSpdz2kProtocol),            //
                     testing::Values(makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field(),
                         std::get<2>(p.param));
    });

}  // namespace spu::mpc::test
