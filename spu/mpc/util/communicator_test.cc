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

#include "spu/mpc/util/communicator.h"

#include "gtest/gtest.h"

#include "spu/mpc/util/simulate.h"

namespace spu::mpc {
namespace {

class CommTest
    : public ::testing::TestWithParam<std::tuple<size_t, FieldType>> {};

TEST_P(CommTest, AllReduce) {
  const Rank kWorldSize = std::get<0>(GetParam());
  const FieldType kField = std::get<1>(GetParam());
  const std::vector<int64_t> kShape = {3, 4};

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    Communicator com(lctx);
    DISPATCH_ALL_FIELDS(kField, "CommTest.AllReduce", [&]() {
      using tensor_t = xt::xarray<ring2k_t>;

      // GIVEN
      const tensor_t a = xt::ones<ring2k_t>(kShape);

      // WHEN
      tensor_t add_a = com.allReduce(ReduceOp::ADD, a, _kName);
      tensor_t xor_a = com.allReduce(ReduceOp::XOR, a, _kName);

      // THEN
      EXPECT_EQ(add_a, xt::ones<ring2k_t>(kShape) * kWorldSize);
      EXPECT_EQ(xor_a, kWorldSize % 2 == 0 ? xt::zeros<ring2k_t>(kShape)
                                           : xt::ones<ring2k_t>(kShape));
    });
  });
}

TEST_P(CommTest, Reduce) {
  const Rank kWorldSize = std::get<0>(GetParam());
  const FieldType kField = std::get<1>(GetParam());
  const std::vector<int64_t> kShape = {3, 4};

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    Communicator com(lctx);
    DISPATCH_ALL_FIELDS(kField, "CommTest.Reduce", [&]() {
      using tensor_t = xt::xarray<ring2k_t>;

      // GIVEN
      const tensor_t a = xt::ones<ring2k_t>(kShape);

      // WHEN
      tensor_t add_a = com.reduce(ReduceOp::ADD, a, 0, _kName);
      tensor_t xor_a = com.reduce(ReduceOp::XOR, a, 0, _kName);

      // THEN
      if (lctx->Rank() == 0) {
        EXPECT_EQ(add_a, xt::ones<ring2k_t>(kShape) * kWorldSize);
        EXPECT_EQ(xor_a, kWorldSize % 2 == 0 ? xt::zeros<ring2k_t>(kShape)
                                             : xt::ones<ring2k_t>(kShape));
      } else {
        EXPECT_EQ(add_a, xt::zeros<ring2k_t>(kShape));
        EXPECT_EQ(xor_a, xt::zeros<ring2k_t>(kShape));
      }
    });
  });
}

TEST_P(CommTest, Rotate) {
  const Rank kWorldSize = std::get<0>(GetParam());
  const FieldType kField = std::get<1>(GetParam());
  const std::vector<int64_t> kShape = {3, 4};

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    Communicator com(lctx);
    DISPATCH_ALL_FIELDS(kField, "CommTest.Reduce", [&]() {
      using tensor_t = xt::xarray<ring2k_t>;

      // GIVEN
      const tensor_t a = xt::ones<ring2k_t>(kShape) * lctx->Rank();

      // WHEN
      tensor_t rotate_a = com.rotate(a, _kName);

      // THEN
      EXPECT_EQ(rotate_a, xt::ones<ring2k_t>(kShape) * lctx->NextRank());
    });
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
