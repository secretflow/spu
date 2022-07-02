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

#include "spu/mpc/util/cexpr.h"

#include "gtest/gtest.h"

namespace spu::mpc::util {

TEST(CExprTest, K) {
  EXPECT_EQ(K()->expr(), "k");
  EXPECT_EQ(K()->eval(FM32, 2), 32);
  EXPECT_EQ(K()->eval(FM64, 3), 64);
  EXPECT_EQ(K()->eval(FM128, 4), 128);
}

TEST(CExprTest, N) {
  EXPECT_EQ(N()->expr(), "n");
  EXPECT_EQ(N()->eval(FM32, 2), 2);
  EXPECT_EQ(N()->eval(FM64, 3), 3);
  EXPECT_EQ(N()->eval(FM128, 4), 4);
}

TEST(CExprTest, Const) {
  auto c = Const(1);
  EXPECT_EQ(c->expr(), "1");
  EXPECT_EQ(c->eval(FM32, 2), 1);
  EXPECT_EQ(c->eval(FM64, 3), 1);
  EXPECT_EQ(c->eval(FM128, 4), 1);
}

TEST(CExprTest, Binary) {
  auto c = 1 + Const(2);
  EXPECT_EQ(c->expr(), "1+2");
  EXPECT_EQ(c->eval(FM32, 2), 3);
  EXPECT_EQ(c->eval(FM64, 3), 3);
  EXPECT_EQ(c->eval(FM128, 4), 3);
}

TEST(CExprTest, Samples) {
  auto c = 2 * (Log(K()) - 1) + 3 * N();
  EXPECT_EQ(c->expr(), "2*(log(k)-1)+3*n");
  EXPECT_EQ(c->eval(FM32, 2), 2 * (std::log2(32) - 1) + 3 * 2);
}

}  // namespace spu::mpc::util
