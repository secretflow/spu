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

#include "libspu/mpc/ref2k/ref2k.h"

#include "libspu/mpc/api_test.h"
#include "libspu/mpc/io_test.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(size_t field) {
  RuntimeConfig conf;
  ProtocolConfig proto_conf;
  proto_conf.set_kind(ProtocolKind::REF2K);
  proto_conf.set_field(field);
  *conf.mutable_protocol() = std::move(proto_conf);
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Ref2kApiTest, ApiTest,
    testing::Combine(testing::Values(makeRef2kProtocol),  //
                     testing::Values(makeConfig(32),      //
                                     makeConfig(64),      //
                                     makeConfig(128)),    //
                     testing::Values(2, 3, 5)),           //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Ref2kIoTest, IoTest,
    testing::Combine(testing::Values(makeRef2kIo),  //
                     testing::Values(2, 3, 5),      //
                     testing::Values(32,            //
                                     64,            //
                                     128)),         //
    [](const testing::TestParamInfo<IoTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });

}  // namespace spu::mpc::test
