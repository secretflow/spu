
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

#include "libspu/mpc/common/pv_gfmp.h"

#include "gtest/gtest.h"

namespace spu::mpc {

TEST(PubGfmpTest, TypeWorks) {
  regPVGfmpTypes();

  {
    Type ty = makeType<PubGfmpTy>(FM32);
    EXPECT_EQ(ty.size(), 4);

    EXPECT_TRUE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_TRUE(ty.isa<GfmpTy>());
    EXPECT_TRUE(ty.as<GfmpTy>()->mp_exp() == 31);
    EXPECT_TRUE(ty.as<GfmpTy>()->p() == (static_cast<uint128_t>(1) << 31) - 1);
    EXPECT_FALSE(ty.isa<Secret>());

    EXPECT_EQ(ty.toString(), "PubGfmp<FM32,31>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
  {
    Type ty = makeType<PubGfmpTy>(FM64);
    EXPECT_EQ(ty.size(), 8);

    EXPECT_TRUE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_TRUE(ty.isa<GfmpTy>());
    EXPECT_TRUE(ty.as<GfmpTy>()->mp_exp() == 61);
    EXPECT_TRUE(ty.as<GfmpTy>()->p() == (static_cast<uint128_t>(1) << 61) - 1);
    EXPECT_FALSE(ty.isa<Secret>());

    EXPECT_EQ(ty.toString(), "PubGfmp<FM64,61>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }

  {
    Type ty = makeType<PubGfmpTy>(FM128);
    EXPECT_EQ(ty.size(), 16);

    EXPECT_TRUE(ty.isa<Public>());
    EXPECT_TRUE(ty.isa<Ring2k>());
    EXPECT_TRUE(ty.isa<Gfp>());
    EXPECT_TRUE(ty.as<GfmpTy>()->mp_exp() == 127);
    EXPECT_TRUE(ty.as<GfmpTy>()->p() == (static_cast<uint128_t>(1) << 127) - 1);
    EXPECT_FALSE(ty.isa<Secret>());

    EXPECT_EQ(ty.toString(), "PubGfmp<FM128,127>");

    EXPECT_EQ(Type::fromString(ty.toString()), ty);
  }
}

}  // namespace spu::mpc
