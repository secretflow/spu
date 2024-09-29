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

#include "libspu/mpc/aby3/ot.h"

#include "gtest/gtest.h"

#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::aby3 {

namespace {

SemanticType GetSemanticType(int64_t field) {
  switch (field) {
    case 32:
      return SE_I32;
    case 64:
      return SE_I64;
    case 128:
      return SE_I128;
  }
  return SE_INVALID;
}

}  // namespace

class OTTest : public ::testing::TestWithParam<
                   std::tuple<size_t, size_t, Ot3::RoleRanks>> {};

TEST_P(OTTest, OT3Party) {
  const int64_t numel = std::get<0>(GetParam());
  const auto field = std::get<1>(GetParam());
  const Ot3::RoleRanks roles = std::get<2>(GetParam());

  Shape shape = {numel};
  MemRef m0(makeType<RingTy>(GetSemanticType(field), field), shape);
  MemRef m1(makeType<RingTy>(GetSemanticType(field), field), shape);

  ring_zeros(m0);
  ring_ones(m1);
  std::vector<uint8_t> choices(numel);
  for (int64_t idx = 0; idx < numel; idx++) {
    choices[idx] = static_cast<uint8_t>(idx % 2);
  }

  utils::simulate(3U, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    Communicator comm(lctx);
    PrgState prg_state(lctx);

    Ot3 ot(field, {numel}, roles, &comm, &prg_state);

    if (comm.getRank() == roles.sender) {
      ot.send(m0, m1);
    } else if (comm.getRank() == roles.receiver) {
      auto mc = ot.recv(choices, GetSemanticType(field));
      EXPECT_TRUE(ring_all_equal(ring_select(choices, m0, m1), mc));
    } else {
      EXPECT_EQ(comm.getRank(), roles.helper);
      ot.help(choices);
    }
  });
}

INSTANTIATE_TEST_SUITE_P(
    OTTestInstances, OTTest,
    testing::Combine(testing::Values(0, 1, 3, 100),
                     testing::Values(32, 64, 128),
                     testing::Values(Ot3::RoleRanks{0, 1, 2},  //
                                     Ot3::RoleRanks{0, 2, 1},  //
                                     Ot3::RoleRanks{1, 0, 2},  //
                                     Ot3::RoleRanks{1, 2, 0},  //
                                     Ot3::RoleRanks{2, 0, 1},  //
                                     Ot3::RoleRanks{2, 1, 0}   //
                                     )),
    [](const testing::TestParamInfo<OTTest::ParamType>& p) {
      const auto roles = std::get<2>(p.param);
      return fmt::format("{}x{}x_{}x{}x{}_", std::get<0>(p.param),
                         std::get<1>(p.param), roles.sender, roles.receiver,
                         roles.helper);
    });

}  // namespace spu::mpc::aby3
