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

#include "spu/mpc/beaver/beaver_he.h"

#include "spu/mpc/beaver/beaver_test.h"

namespace spu::mpc {

INSTANTIATE_TEST_SUITE_P(
    BeaverHETest, BeaverTest,
    testing::Combine(
        testing::Values([](const std::shared_ptr<yasl::link::Context>& lctx) {
          (void)lctx;
          return std::make_unique<BeaverHE>(lctx);
        }),
        testing::Values(2),  // 2-party
        testing::Values(FieldType::FM64, FieldType::FM32), testing::Values(1)),
    [](const testing::TestParamInfo<BeaverTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });
}
