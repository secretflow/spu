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

#include "libspu/mpc/semi2k/type.h"

#include "gtest/gtest.h"

namespace spu::mpc::semi2k {

TEST(ArithShareTy, Simple) {
  registerTypes();
  {
    Type ty = makeType<ArithShareTy>(SE_I32, 32);
    EXPECT_EQ(ty.size(), 4);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<BaseRingType>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<ArithShare>());
    EXPECT_FALSE(ty.isa<BoolShare>());

    EXPECT_EQ(ty.toString(), "semi2k.ArithShare<SE_I32,32>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

TEST(BoolShareTy, Simple) {
  // Semi2k::BShr constructor with field.
  {
    Type ty = makeType<BoolShareTy>(SE_I128, ST_128, 128);
    EXPECT_EQ(ty.size(), 16);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<BaseRingType>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_FALSE(ty.isa<ArithShare>());
    EXPECT_TRUE(ty.isa<BoolShare>());

    EXPECT_EQ(ty.toString(), "semi2k.BoolShare<SE_I128,ST_128,128>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }

  // Semi2k::BShr constructor with field and nbits.
  {
    Type ty = makeType<BoolShareTy>(SE_I128, ST_128, 7);
    EXPECT_EQ(ty.size(), 16);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<BaseRingType>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_FALSE(ty.isa<ArithShare>());
    EXPECT_TRUE(ty.isa<BoolShare>());

    EXPECT_EQ(ty.toString(), "semi2k.BoolShare<SE_I128,ST_128,7>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

}  // namespace spu::mpc::semi2k
