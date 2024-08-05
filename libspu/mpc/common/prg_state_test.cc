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

#include "libspu/mpc/common/prg_state.h"

#include "gtest/gtest.h"
#include "yacl/link/link.h"

#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc {

TEST(PrgStateTest, Fork) {
  const size_t npc = 3;

  std::array<std::array<int, 2>, 3> data0;
  std::array<std::array<int, 2>, 3> data1;
  std::array<std::array<int, 2>, 3> data2;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const size_t rank = lctx->Rank();
    auto root = std::make_unique<PrgState>(lctx);

    auto sub1 = root->fork();
    auto sub2 = root->fork();
    auto sub1_sub1 = sub1->fork();

    dynamic_cast<PrgState*>(sub1.get())
        ->fillPrssPair(absl::MakeSpan(data0[rank].data(), 1),
                       absl::MakeSpan(&data0[rank][1], 1));
    dynamic_cast<PrgState*>(sub2.get())
        ->fillPrssPair(absl::MakeSpan(data1[rank].data(), 1),
                       absl::MakeSpan(&data1[rank][1], 1));
    dynamic_cast<PrgState*>(sub1_sub1.get())
        ->fillPrssPair(absl::MakeSpan(data2[rank].data(), 1),
                       absl::MakeSpan(&data2[rank][1], 1));
    yacl::link::Barrier(lctx, "_");

    for (size_t idx = 0; idx < npc; idx++) {
      EXPECT_EQ(data0[idx][1], data0[(idx + 1) % npc][0]);
    }

    for (size_t idx = 0; idx < npc; idx++) {
      EXPECT_EQ(data1[idx][1], data1[(idx + 1) % npc][0]);
    }

    for (size_t idx = 0; idx < npc; idx++) {
      EXPECT_EQ(data2[idx][1], data2[(idx + 1) % npc][0]);
    }
  });
}

}  // namespace spu::mpc
