
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

#include "libspu/mpc/common/pv2k.h"

#include "gtest/gtest.h"

namespace spu::mpc {

TEST(Pub2kTest, TypeWorks) {
  regPV2kTypes();

  {
    Type ty = makeType<Pub2kTy>(FM32);
    EXPECT_EQ(ty.size(), 4);

    EXPECT_TRUE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Secret>());

    EXPECT_EQ(ty.toString(), "Pub2k<FM32>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }

  {
    Type ty = makeType<Pub2kTy>(FM128);
    EXPECT_EQ(ty.size(), 16);

    EXPECT_TRUE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Secret>());

    EXPECT_EQ(ty.toString(), "Pub2k<FM128>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

TEST(Priv2kTest, TypeWorks) {
  regPV2kTypes();

  {
    Type ty = makeType<Priv2kTy>(FM32, -1);
    EXPECT_EQ(ty.size(), 4);

    EXPECT_TRUE(ty.isa<Private>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Secret>());

    EXPECT_EQ(ty.toString(), "Priv2k<FM32,-1>");

    const Type ty1 = Type::fromString(ty.toString());
    EXPECT_EQ(ty1.as<Priv2kTy>()->owner(), -1);
    EXPECT_EQ(ty1, ty);
  }

  {
    Type ty = makeType<Priv2kTy>(FM128, 1);
    EXPECT_EQ(ty.size(), 16);

    EXPECT_TRUE(ty.isa<Private>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Secret>());

    EXPECT_EQ(ty.toString(), "Priv2k<FM128,1>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

}  // namespace spu::mpc
