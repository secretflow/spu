// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/utils/simulate.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::utils {

TEST(Simulate, Works) {
  auto result = simulate(
      3,
      [](const std::shared_ptr<yacl::link::Context>& lctx, int x) {
        return 1 + lctx->Rank();
      },
      1);

  EXPECT_THAT(result, testing::ElementsAre(1, 2, 3));
}

TEST(Simulate, Throw) {
  EXPECT_THROW(
      {
        simulate(
            3,
            [](const std::shared_ptr<yacl::link::Context>& lctx, int x) {
              if (lctx->Rank() == 2) {
                SPU_THROW("error");
              }
              return 1 + lctx->Rank();
            },
            1);
      },
      std::exception);
}

}  // namespace spu::mpc::utils
