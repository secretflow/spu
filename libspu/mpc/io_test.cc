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

#include "libspu/mpc/io_test.h"

#include "gtest/gtest.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

const size_t kNumel = 7;

TEST_P(IoTest, MakePublicAndReconstruct) {
  const auto create_io = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  auto io = create_io(field, npc);

  auto raw = ring_rand(field, kNumel);
  auto shares = io->toShares(raw, VIS_PUBLIC);
  auto result = io->fromShares(shares);

  EXPECT_TRUE(ring_all_equal(raw, result));
}

TEST_P(IoTest, MakeSecretAndReconstruct) {
  const auto create_io = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  auto io = create_io(field, npc);

  auto raw = ring_rand(field, kNumel);
  auto shares = io->toShares(raw, VIS_SECRET);
  auto result = io->fromShares(shares);

  EXPECT_TRUE(ring_all_equal(raw, result));
}

TEST_P(IoTest, MakeSecretAndReconstructWithInvalidOwnerRank) {
  const auto create_io = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  auto io = create_io(field, npc);

  auto raw = ring_rand(field, kNumel);
  auto shares = io->toShares(raw, VIS_SECRET, -1);
  auto result = io->fromShares(shares);

  EXPECT_TRUE(ring_all_equal(raw, result));
}

TEST_P(IoTest, MakeSecretAndReconstructWithValidOwnerRank) {
  const auto create_io = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  auto io = create_io(field, npc);

  auto raw = ring_rand(field, kNumel);
  auto shares = io->toShares(raw, VIS_SECRET, 0);
  auto result = io->fromShares(shares);

  EXPECT_TRUE(ring_all_equal(raw, result));
}

}  // namespace spu::mpc
