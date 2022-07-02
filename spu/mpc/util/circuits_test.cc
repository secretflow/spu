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

#include "spu/mpc/util/circuits.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu::mpc {
namespace {

template <typename C>
CircuitBasicBlock<C> makeVectorCBB() {
  CircuitBasicBlock<C> cbb;
  cbb.num_bits = sizeof(typename C::value_type) * 8;
  cbb.init_like = [](C const& in, uint64_t init) -> C {
    return C(in.size(), init);
  };
  cbb._xor = [](C const& lhs, C const& rhs) -> C {
    C res;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(res),
                   std::bit_xor<>());
    return res;
  };
  cbb._and = [](C const& lhs, C const& rhs) -> C {
    C res;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(res),
                   std::bit_and<>());
    return res;
  };
  cbb.lshift = [](C const& x, size_t bits) -> C {
    C res;
    std::transform(x.begin(), x.end(), std::back_inserter(res),
                   [&](const auto& e) { return e << bits; });
    return res;
  };
  cbb.rshift = [](C const& x, size_t bits) -> C {
    C res;
    std::transform(x.begin(), x.end(), std::back_inserter(res),
                   [&](const auto& e) { return e >> bits; });
    return res;
  };
  return cbb;
}

}  // namespace

TEST(KoggleStoneAdder, Scalar) {
  EXPECT_EQ(KoggleStoneAdder(42, 17), 42 + 17);
  EXPECT_EQ(KoggleStoneAdder(0xFFFFFFFFU, 0x00000000U), 0xFFFFFFFFU);
  EXPECT_EQ(KoggleStoneAdder(0xFFFFFFFFU, 0x00000001U), 0x00000000U);
  EXPECT_EQ(KoggleStoneAdder(0xFFFFFFFFU, 0x00010000U), 0x0000FFFFU);
  EXPECT_EQ(KoggleStoneAdder(0xFFFFFFFFU, 0x10000000U), 0x0FFFFFFFU);
  EXPECT_EQ(KoggleStoneAdder(0xFFFFFFFFU, 0xFFFFFFFFU), 0xFFFFFFFEU);
  EXPECT_EQ(KoggleStoneAdder(0xFFFF0000U, 0x0000FFFFU), 0xFFFFFFFFU);
  EXPECT_EQ(KoggleStoneAdder(0x0000FFFFU, 0xFFFF0000U), 0xFFFFFFFFU);
  EXPECT_EQ(KoggleStoneAdder(0x0000FFFFU, 0xFFFF0001U), 0x00000000U);
  EXPECT_EQ(KoggleStoneAdder(0x0100FFFFU, 0xFFFF0000U), 0x00FFFFFFU);
  EXPECT_EQ(KoggleStoneAdder(0x0F0F0F0FU, 0xF0F0F0F0U), 0xFFFFFFFFU);
  EXPECT_EQ(KoggleStoneAdder(0xF0F0F0F0U, 0x0F0F0F0FU), 0xFFFFFFFFU);
}

TEST(KoggleStoneAdder, Vectorized) {
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> cases = {
      {0xFFFFFFFFU, 0x00000000U, 0xFFFFFFFFU},  //
      {0xFFFFFFFFU, 0x00000001U, 0x00000000U},  //
      {0xFFFFFFFFU, 0x00010000U, 0x0000FFFFU},  //
      {0xFFFFFFFFU, 0x10000000U, 0x0FFFFFFFU},  //
      {0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFEU},  //
      {0xFFFF0000U, 0x0000FFFFU, 0xFFFFFFFFU},  //
      {0x0000FFFFU, 0xFFFF0000U, 0xFFFFFFFFU},  //
      {0x0000FFFFU, 0xFFFF0001U, 0x00000000U},  //
      {0x0100FFFFU, 0xFFFF0000U, 0x00FFFFFFU},  //
      {0x0F0F0F0FU, 0xF0F0F0F0U, 0xFFFFFFFFU},  //
      {0xF0F0F0F0U, 0x0F0F0F0FU, 0xFFFFFFFFU},  //
  };

  using Vector = std::vector<uint32_t>;
  Vector lhs, rhs, ans;
  std::transform(cases.begin(), cases.end(), std::back_inserter(lhs),
                 [](auto const& t) { return std::get<0>(t); });
  std::transform(cases.begin(), cases.end(), std::back_inserter(rhs),
                 [](auto const& t) { return std::get<1>(t); });
  std::transform(cases.begin(), cases.end(), std::back_inserter(ans),
                 [](auto const& t) { return std::get<2>(t); });

  Vector res = KoggleStoneAdder(lhs, rhs, makeVectorCBB<Vector>());
  EXPECT_THAT(res, testing::ElementsAreArray(ans.begin(), ans.end()));
}

TEST(CarryOut, Scalar) {
  EXPECT_EQ(CarryOut(0xFFFFFFFFU, 0x00000000U), 0);
  EXPECT_EQ(CarryOut(0xFFFFFFFFU, 0x00000001U), 1);
  EXPECT_EQ(CarryOut(0xFFFFFFFFU, 0x00010000U), 1);
  EXPECT_EQ(CarryOut(0xFFFFFFFFU, 0x10000000U), 1);
  EXPECT_EQ(CarryOut(0xFFFFFFFFU, 0xFFFFFFFFU), 1);
  EXPECT_EQ(CarryOut(0xFFFF0000U, 0x0000FFFFU), 0);
  EXPECT_EQ(CarryOut(0x0000FFFFU, 0xFFFF0000U), 0);
  EXPECT_EQ(CarryOut(0x0000FFFFU, 0xFFFF0001U), 1);
  EXPECT_EQ(CarryOut(0x0100FFFFU, 0xFFFF0000U), 1);
  EXPECT_EQ(CarryOut(0x0F0F0F0FU, 0xF0F0F0F0U), 0);
  EXPECT_EQ(CarryOut(0xF0F0F0F0U, 0x0F0F0F0FU), 0);
  EXPECT_EQ(CarryOut(0xF0F0F0F1U, 0x0F0F0F0FU), 1);
  EXPECT_EQ(CarryOut(0xF0F1F0F0U, 0x0F0F0F0FU), 1);
}

TEST(CarryOut, Vectorized) {
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> cases = {
      {0xFFFFFFFF, 0x00000000, 0},  //
      {0xFFFFFFFF, 0x00000001, 1},  //
      {0xFFFFFFFF, 0x00010000, 1},  //
      {0xFFFFFFFF, 0x10000000, 1},  //
      {0xFFFFFFFF, 0xFFFFFFFF, 1},  //
      {0xFFFF0000, 0x0000FFFF, 0},  //
      {0x0000FFFF, 0xFFFF0000, 0},  //
      {0x0000FFFF, 0xFFFF0001, 1},  //
      {0x0100FFFF, 0xFFFF0000, 1},  //
      {0x0F0F0F0F, 0xF0F0F0F0, 0},  //
      {0xF0F0F0F0, 0x0F0F0F0F, 0},  //
      {0xF0F0F0F1, 0x0F0F0F0F, 1},  //
      {0xF0F1F0F0, 0x0F0F0F0F, 1},  //
  };

  using Vector = std::vector<uint32_t>;
  Vector lhs, rhs, ans;
  std::transform(cases.begin(), cases.end(), std::back_inserter(lhs),
                 [](auto const& t) { return std::get<0>(t); });
  std::transform(cases.begin(), cases.end(), std::back_inserter(rhs),
                 [](auto const& t) { return std::get<1>(t); });
  std::transform(cases.begin(), cases.end(), std::back_inserter(ans),
                 [](auto const& t) { return std::get<2>(t); });

  Vector res = CarryOut(lhs, rhs, makeVectorCBB<Vector>());
  EXPECT_THAT(res, testing::ElementsAreArray(ans.begin(), ans.end()));
}

}  // namespace spu::mpc
