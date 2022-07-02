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

#include "spu/mpc/semi2k/protocol.h"

#include "spu/mpc/common/abprotocol_test.h"
#include "spu/mpc/compute_test.h"

namespace spu::mpc::test {

INSTANTIATE_TEST_SUITE_P(
    Semi2kComputeTest, ComputeTest,
    testing::Combine(testing::Values(makeSemi2kProtocol),  //
                     testing::Values(2, 3, 5),             //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<ComputeTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2kArithmeticTest, ArithmeticTest,
    testing::Combine(testing::Values(makeSemi2kProtocol),  //
                     testing::Values(2, 3, 5),             //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2kBooleanTest, BooleanTest,
    testing::Combine(testing::Values(makeSemi2kProtocol),  //
                     testing::Values(2, 3, 5),             //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2kBooleanTest, ConversionTest,
    testing::Combine(testing::Values(makeSemi2kProtocol),  //
                     testing::Values(2, 3, 5),             //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });

}  // namespace spu::mpc::test
