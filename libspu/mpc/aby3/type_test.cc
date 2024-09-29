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

#include "libspu/mpc/aby3/type.h"

#include "gtest/gtest.h"

namespace spu::mpc::aby3 {

TEST(ArithShareTy, Simple) {
  registerTypes();
  {
    Type ty = makeType<ArithShareTy>(SE_I32, 32);
    EXPECT_EQ(ty.size(), 4 * 2);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<RingTy>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<ArithShare>());
    EXPECT_FALSE(ty.isa<BoolShare>());

    EXPECT_EQ(ty.toString(), "aby3.AShr<SE_I32,32>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

TEST(BoolShareTy, Simple) {
  Type ty = makeType<BoolShareTy>();
  // aby3::BShr constructor with field and nbits.
  {
    Type ty = makeType<BoolShareTy>(SE_I8, ST_8, 7);
    EXPECT_EQ(ty.size(), SizeOf(ST_8) * 2);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_FALSE(ty.isa<ArithShare>());
    EXPECT_TRUE(ty.isa<BoolShare>());

    EXPECT_EQ(ty.toString(), "aby3.BShr<SE_I8,7>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);

    // clone
    Type cty = ty;  // NOLINT: Test clone
    EXPECT_EQ(cty, ty);
    EXPECT_TRUE(cty.isa<Secret>());
    EXPECT_FALSE(cty.isa<Public>());
    EXPECT_FALSE(cty.isa<ArithShare>());
    EXPECT_TRUE(cty.isa<BoolShare>());

    EXPECT_EQ(cty.toString(), "aby3.BShr<SE_I8,7>");
  }
}

}  // namespace spu::mpc::aby3
