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

#include "libspu/mpc/securenn/type.h"

#include "gtest/gtest.h"

namespace spu::mpc::securenn {

TEST(AShrTy, Simple) {
  registerTypes();
  {
    Type ty = makeType<AShrTy>(FM32);
    EXPECT_EQ(ty.size(), 4);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<AShare>());
    EXPECT_FALSE(ty.isa<BShare>());

    EXPECT_EQ(ty.toString(), "securenn.AShr<FM32>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

TEST(BShrTy, Simple) {
  Type ty = makeType<BShrTy>();
  // Securenn::BShr constructor with field.
  {
    Type ty = makeType<BShrTy>(FM128);
    EXPECT_EQ(ty.size(), 16);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_FALSE(ty.isa<AShare>());
    EXPECT_TRUE(ty.isa<BShare>());

    EXPECT_EQ(ty.toString(), "securenn.BShr<FM128,128>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }

  // Securenn::BShr constructor with field and nbits.
  {
    Type ty = makeType<BShrTy>(FM128, 7);
    EXPECT_EQ(ty.size(), 16);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_FALSE(ty.isa<AShare>());
    EXPECT_TRUE(ty.isa<BShare>());

    EXPECT_EQ(ty.toString(), "securenn.BShr<FM128,7>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

}  // namespace spu::mpc::securenn
