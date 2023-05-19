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

#include "libspu/core/cexpr.h"

#include <cmath>

#include "gtest/gtest.h"

namespace spu::ce {

TEST(CExprTest, K) {
  EXPECT_EQ(K()->expr(), "K");
  EXPECT_EQ(K()->eval({{"K", 32}, {"N", 2}}), 32);
  EXPECT_EQ(K()->eval({{"K", 64}, {"N", 3}}), 64);
  EXPECT_EQ(K()->eval({{"K", 128}, {"N", 4}}), 128);
}

TEST(CExprTest, N) {
  EXPECT_EQ(N()->expr(), "N");
  EXPECT_EQ(N()->eval({{"K", 32}, {"N", 2}}), 2);
  EXPECT_EQ(N()->eval({{"K", 64}, {"N", 3}}), 3);
  EXPECT_EQ(N()->eval({{"K", 128}, {"N", 4}}), 4);
}

TEST(CExprTest, Const) {
  auto c = Const(1);
  EXPECT_EQ(c->expr(), "1");
  EXPECT_EQ(c->eval({{"K", 32}, {"N", 2}}), 1);
  EXPECT_EQ(c->eval({{"K", 64}, {"N", 3}}), 1);
  EXPECT_EQ(c->eval({{"K", 128}, {"N", 4}}), 1);
}

TEST(CExprTest, Binary) {
  auto c = 1 + Const(2);
  EXPECT_EQ(c->expr(), "1+2");
  EXPECT_EQ(c->eval({{"K", 32}, {"N", 2}}), 3);
  EXPECT_EQ(c->eval({{"K", 64}, {"N", 3}}), 3);
  EXPECT_EQ(c->eval({{"K", 128}, {"N", 4}}), 3);
}

TEST(CExprTest, Samples) {
  auto c = 2 * (Log(K()) - 1) + 3 * N();
  EXPECT_EQ(c->expr(), "2*(log(K)-1)+3*N");
  EXPECT_EQ(c->eval({{"K", 32}, {"N", 2}}), 2 * (std::log2(32) - 1) + 3 * 2);
}

}  // namespace spu::ce
