// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/experimental/shamir/type.h"

#include "gtest/gtest.h"

namespace spu::mpc::shamir {

TEST(AShrTy, Simple) {
  registerTypes();
  {
    Type ty = makeType<AShrTy>(FM32);
    EXPECT_EQ(ty.size(), 4);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<AShare>());
    EXPECT_TRUE(ty.isa<GfmpTy>());
    EXPECT_TRUE(ty.as<GfmpTy>()->mp_exp() == 31);
    EXPECT_TRUE(ty.as<GfmpTy>()->p() ==
                (static_cast<uint128_t>(1) << ty.as<GfmpTy>()->mp_exp()) - 1);

    EXPECT_EQ(ty.toString(), "shamir.AShr<FM32,31>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

TEST(BShrTy, Simple) {
  {
    Type ty = makeType<BShrTy>(FM128);
    EXPECT_EQ(ty.size(), 16 * 128);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_FALSE(ty.isa<AShare>());
    EXPECT_TRUE(ty.isa<BShare>());

    EXPECT_EQ(ty.toString(), "shamir.BShr<FM128,128>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }

  // Semi2k::BShr constructor with field and nbits.
  {
    Type ty = makeType<BShrTy>(FM128, 7);
    EXPECT_EQ(ty.size(), 16 * 7);

    EXPECT_TRUE(ty.isa<Secret>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_FALSE(ty.isa<Public>());
    EXPECT_FALSE(ty.isa<AShare>());
    EXPECT_TRUE(ty.isa<BShare>());

    EXPECT_EQ(ty.toString(), "shamir.BShr<FM128,7>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

}  // namespace spu::mpc::shamir
