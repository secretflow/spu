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

#include "libspu/mpc/common/communicator.h"

#include <utility>

#include "gtest/gtest.h"

#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc {
namespace {

class CommTest
    : public ::testing::TestWithParam<std::tuple<size_t, FieldType>> {};

TEST_P(CommTest, AllReduce) {
  const Rank kWorldSize = std::get<0>(GetParam());
  const FieldType kField = std::get<1>(GetParam());
  const int64_t kNumel = 1000;

  std::vector<ArrayRef> xs(kWorldSize);
  ArrayRef sum_x = ring_zeros(kField, kNumel);
  ArrayRef xor_x = ring_zeros(kField, kNumel);
  for (size_t idx = 0; idx < kWorldSize; idx++) {
    xs[idx] = ring_rand(kField, kNumel);
    ring_add_(sum_x, xs[idx]);
    ring_xor_(xor_x, xs[idx]);
  }

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    Communicator com(std::move(lctx));
    // WHEN
    auto sum_r = com.allReduce(ReduceOp::ADD, xs[com.getRank()], "_");
    auto xor_r = com.allReduce(ReduceOp::XOR, xs[com.getRank()], "_");

    // THEN
    EXPECT_TRUE(ring_all_equal(sum_r, sum_x));
    EXPECT_TRUE(ring_all_equal(xor_r, xor_x));
  });
}

TEST_P(CommTest, Reduce) {
  const Rank kWorldSize = std::get<0>(GetParam());
  const FieldType kField = std::get<1>(GetParam());
  const int64_t kNumel = 1000;

  std::vector<ArrayRef> xs(kWorldSize);
  ArrayRef sum_x = ring_zeros(kField, kNumel);
  ArrayRef xor_x = ring_zeros(kField, kNumel);
  for (size_t idx = 0; idx < kWorldSize; idx++) {
    xs[idx] = ring_rand(kField, kNumel);
    ring_add_(sum_x, xs[idx]);
    ring_xor_(xor_x, xs[idx]);
  }

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    Communicator com(std::move(lctx));

    for (size_t root = 0; root < kWorldSize; root++) {
      // WHEN
      auto sum_r = com.reduce(ReduceOp::ADD, xs[com.getRank()], root, "_");
      auto xor_r = com.reduce(ReduceOp::XOR, xs[com.getRank()], root, "_");

      // THEN
      if (com.getRank() == root) {
        EXPECT_TRUE(ring_all_equal(sum_r, sum_x));
        EXPECT_TRUE(ring_all_equal(xor_r, xor_x));
      } else {
        EXPECT_TRUE(ring_all_equal(sum_r, xs[com.getRank()]));
        EXPECT_TRUE(ring_all_equal(xor_r, xs[com.getRank()]));
      }
    }
  });
}

TEST_P(CommTest, Rotate) {
  const Rank kWorldSize = std::get<0>(GetParam());
  const FieldType kField = std::get<1>(GetParam());
  const int64_t kNumel = 1000;

  std::vector<ArrayRef> xs(kWorldSize);
  for (size_t idx = 0; idx < kWorldSize; idx++) {
    xs[idx] = ring_rand(kField, kNumel);
  }

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    Communicator com(std::move(lctx));
    // WHEN
    auto r = com.rotate(xs[com.getRank()], "_");

    // THEN
    EXPECT_TRUE(ring_all_equal(r, xs[(com.getRank() + 1) % kWorldSize]));
  });
}

INSTANTIATE_TEST_SUITE_P(
    CommTestInstances, CommTest,
    testing::Combine(testing::Values(4, 3, 2),
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<CommTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

}  // namespace
}  // namespace spu::mpc
