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

#include "libspu/mpc/utils/circuits.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yacl/base/int128.h"

namespace spu::mpc {
namespace {

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
CircuitBasicBlock<T> makeScalarCBB() {
  CircuitBasicBlock<T> cbb;
  cbb._xor = [](T const& lhs, T const& rhs) -> T { return lhs ^ rhs; };
  cbb._and = [](T const& lhs, T const& rhs) -> T { return lhs & rhs; };
  cbb.lshift = [](T const& x, size_t bits) -> T { return x << bits; };
  cbb.rshift = [](T const& x, size_t bits) -> T { return x >> bits; };
  cbb.init_like = [](T const&, uint128_t init) -> T {
    return static_cast<T>(init);
  };
  cbb.set_nbits = [](T& x, size_t nbits) {};
  return cbb;
}

template <typename C, typename T = typename C::value_type>
CircuitBasicBlock<C> makeVectorCBB() {
  CircuitBasicBlock<C> cbb;
  cbb.init_like = [](C const& in, uint128_t init) -> C {
    return C(in.size(), static_cast<T>(init));
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
  cbb.set_nbits = [](C& x, size_t nbits) -> void {};
  return cbb;
}

}  // namespace

std::vector<std::vector<uint32_t>> kU32Add = {
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

TEST(KoggeStoneAdder, Scalar) {
  using T = uint32_t;
  const auto cbb = makeScalarCBB<T>();
  const size_t nbits = sizeof(T) * 8;

  for (auto item : kU32Add) {
    EXPECT_EQ(kogge_stone(cbb, item[0], item[1], nbits), item[0] + item[1]);
  }
}

TEST(KoggeStoneAdder, Vectorized) {
  using T = uint32_t;
  const size_t nbits = sizeof(T) * 8;
  using VT = std::vector<T>;

  std::vector<VT> args(3);
  for (auto item : kU32Add) {
    for (size_t idx = 0; idx < 3; idx++) {
      args[idx].push_back(item[idx]);
    }
  }

  auto cbb = makeVectorCBB<VT>();
  auto r0 = kogge_stone(cbb, args[0], args[1], nbits);
  EXPECT_THAT(r0, testing::ElementsAreArray(args[2].begin(), args[2].end()));
}

TEST(SklanskyAdder, Scalar) {
  using T = uint32_t;
  const auto cbb = makeScalarCBB<T>();
  const size_t nbits = sizeof(T) * 8;

  for (auto item : kU32Add) {
    EXPECT_EQ(sklansky(cbb, item[0], item[1], nbits), item[2]);
  }
}

TEST(SklanskyAdder, Vectorized) {
  using T = uint32_t;
  const size_t nbits = sizeof(T) * 8;
  using VT = std::vector<T>;

  std::vector<VT> args(3);
  for (auto item : kU32Add) {
    for (size_t idx = 0; idx < 3; idx++) {
      args[idx].push_back(item[idx]);
    }
  }

  auto cbb = makeVectorCBB<VT>();
  auto r1 = sklansky(cbb, args[0], args[1], nbits);
  EXPECT_THAT(r1, testing::ElementsAreArray(args[2].begin(), args[2].end()));
}

TEST(UtilTest, Works) {
  using T = uint32_t;
  const auto cbb = makeScalarCBB<T>();

  ASSERT_EQ(odd_even_split(cbb, 0xAAAAAAAAU, 32), 0xFFFF0000U);
  ASSERT_EQ(odd_even_split(cbb, 0xAAAAAAAAU, 16), 0xFF00FF00U);
  ASSERT_EQ(odd_even_split(cbb, 0x0000AAAAU, 16), 0x0000FF00U);
  ASSERT_EQ(odd_even_split(cbb, 0x00000AAAU, 12), 0x00000FC0U);
  ASSERT_EQ(odd_even_split(cbb, 0x000000AAU, 8L), 0x000000F0U);
  ASSERT_EQ(odd_even_split(cbb, 0x0000002AU, 6L), 0x00000038U);
  ASSERT_EQ(odd_even_split(cbb, 0x00000015U, 6L), 0x00000007U);
  ASSERT_EQ(odd_even_split(cbb, 0x0000000AU, 4L), 0x0000000CU);
  ASSERT_EQ(odd_even_split(cbb, 0x00000002U, 2L), 0x00000002U);
  ASSERT_EQ(odd_even_split(cbb, 0x00000001U, 2L), 0x00000001U);
  ASSERT_EQ(odd_even_split(cbb, 0x000002AAU, 10), 0x000003E0U);
  ASSERT_EQ(odd_even_split(cbb, 0x55555555U, 32), 0x0000FFFFU);
}

std::vector<std::vector<uint32_t>> kU32Carry = {
    {0xFFFFFFFFU, 0x00000000U, 0, 0xFFFFFFFFU},  //
    {0xFFFFFFFFU, 0x00000001U, 1, 0x00000000U},  //
    {0xFFFFFFFFU, 0x00010000U, 1, 0x0000FFFFU},  //
    {0xFFFFFFFFU, 0x10000000U, 1, 0x0FFFFFFFU},  //
    {0xFFFFFFFFU, 0xFFFFFFFFU, 1, 0xFFFFFFFEU},  //
    {0xFFFF0000U, 0x0000FFFFU, 0, 0xFFFFFFFFU},  //
    {0x0000FFFFU, 0xFFFF0000U, 0, 0xFFFFFFFFU},  //
    {0x0000FFFFU, 0xFFFF0001U, 1, 0x00000000U},  //
    {0x0100FFFFU, 0xFFFF0000U, 1, 0x00FFFFFFU},  //
    {0x0F0F0F0FU, 0xF0F0F0F0U, 0, 0xFFFFFFFFU},  //
    {0xF0F0F0F0U, 0x0F0F0F0FU, 0, 0xFFFFFFFFU},  //
    {0xF0F0F0F1U, 0x0F0F0F0FU, 1, 0x00000001U},  //
    {0xF0F1F0F0U, 0x0F0F0F0FU, 1, 0x00010000U},  //

};

TEST(CarryOutTest, Scalar) {
  using T = uint32_t;
  const auto cbb = makeScalarCBB<T>();
  const size_t nbits = sizeof(T) * 8;

  for (auto item : kU32Carry) {
    const auto x = item[0];
    const auto y = item[1];
    for (size_t k = 1; k < nbits; k++) {
      EXPECT_EQ(carry_out(cbb, x, y, k), ((((x + y) ^ (x ^ y)) >> k) & 1))
          << k << std::hex << " " << x << " " << y;
    }
    ASSERT_EQ(carry_out(cbb, x, y, nbits), item[2])
        << std::hex << x << " " << y;
  }
}

TEST(CarryOutTest, Vectorized) {
  using T = uint32_t;
  const size_t nbits = sizeof(T) * 8;
  using VT = std::vector<T>;

  std::vector<VT> args(3);
  for (auto item : kU32Carry) {
    for (size_t idx = 0; idx < 3; idx++) {
      args[idx].push_back(item[idx]);
    }
  }

  auto cbb = makeVectorCBB<VT>();
  auto r0 = carry_out(cbb, args[0], args[1], nbits);
  EXPECT_THAT(r0, testing::ElementsAreArray(args[2].begin(), args[2].end()));

  for (size_t k = 1; k < nbits; k++) {
    VT expected;
    for (size_t idx = 0; idx < args[0].size(); idx++) {
      const auto x = args[0][idx];
      const auto y = args[1][idx];
      auto c_xy = (((x + y) ^ (x ^ y)) >> k) & 1;
      expected.push_back(c_xy);
    }
    auto ck = carry_out(cbb, args[0], args[1], k);
    EXPECT_THAT(ck,
                testing::ElementsAreArray(expected.begin(), expected.end()));
  }
}

}  // namespace spu::mpc
