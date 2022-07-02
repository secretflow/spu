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

#include "spu/core/vectorize.h"

#include <list>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(SimdTest, SimdTrait) {
  struct NotQuiteLike {
    void begin();
    void end();
  };
  EXPECT_EQ(detail::is_container_like_v<size_t>, false);
  EXPECT_EQ(detail::is_container_like_v<std::vector<size_t>>, true);
  EXPECT_EQ(detail::is_container_like_v<std::string>, true);
  EXPECT_EQ(detail::is_container_like_v<std::list<char>>, true);
  EXPECT_EQ(detail::is_container_like_v<NotQuiteLike>, false);

  EXPECT_EQ(hasSimdTrait<size_t>::value, false);
  EXPECT_EQ(hasSimdTrait<std::vector<size_t>>::value, true);
  EXPECT_EQ(hasSimdTrait<std::string>::value, true);
  EXPECT_EQ(hasSimdTrait<std::list<char>>::value, true);
  EXPECT_EQ(hasSimdTrait<NotQuiteLike>::value, false);
}

using Vector = std::vector<int>;
Vector operator+(const Vector& a, const Vector& b) {
  Vector c;
  std::transform(a.begin(), a.end(),     // first
                 b.begin(),              // second
                 std::back_inserter(c),  // output
                 std::plus<>());
  return c;
}

TEST(VectorizeTest, Vector) {
  std::vector<Vector> a = {{1, 2}, {3, 4, 5}, {6}};
  std::vector<Vector> b = {{1, 2}, {3, 4, 5}, {6}};

  size_t num_calls = 0;
  auto vector_add = [&](const Vector& a, const Vector& b) -> Vector {
    num_calls++;
    return a + b;
  };

  std::vector<Vector> c;
  vectorize(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c),
            vector_add);

  EXPECT_EQ(num_calls, 1);
  EXPECT_EQ(c.size(), a.size());
  EXPECT_THAT(c[0], testing::ElementsAre(2, 4));
  EXPECT_THAT(c[1], testing::ElementsAre(6, 8, 10));
  EXPECT_THAT(c[2], testing::ElementsAre(12));

  ///
  {
    Vector x = {1, 2};
    Vector y = {3, 4, 5};
    num_calls = 0;
    auto result = vectorize({x, y}, {x, y}, vector_add);
    EXPECT_EQ(num_calls, 1);
    EXPECT_EQ(result.size(), 2);
    EXPECT_THAT(c[0], testing::ElementsAre(2, 4));
    EXPECT_THAT(c[1], testing::ElementsAre(6, 8, 10));
  }
}

TEST(VectorizeTest, Reduce) {
  std::vector<Vector> a = {
      {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7},
  };

  size_t num_calls = 0;
  auto vector_add = [&](const Vector& a, const Vector& b) -> Vector {
    num_calls++;
    return a + b;
  };

  auto res = vectorizedReduce(a.begin(), a.end(), vector_add);

  EXPECT_EQ(num_calls, std::ceil(std::log2(a.size())));
  EXPECT_EQ(res.size(), 2);
  EXPECT_THAT(res, testing::ElementsAre(28, 28));
}

}  // namespace spu
