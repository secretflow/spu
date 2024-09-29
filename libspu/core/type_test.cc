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

#include "libspu/core/type.h"

#include "gtest/gtest.h"

namespace spu {

TEST(TypeTest, VoidTy) {
  Type va;
  //
  EXPECT_FALSE(va.isa<RingTy>());
  EXPECT_TRUE(va.isa<VoidTy>());
  EXPECT_EQ(va.size(), 0);
  EXPECT_EQ(va.toString(), "Void<>");
  EXPECT_EQ(Type::fromString(va.toString()), va);

  Type vb = va;  // NOLINT: Test copy ctor
  EXPECT_FALSE(vb.isa<RingTy>());
  EXPECT_EQ(vb.size(), 0);
  EXPECT_EQ(vb.toString(), "Void<>");

  EXPECT_EQ(va, vb);
  EXPECT_EQ(vb, va);
}

TEST(TypeTest, RingTy) {
  Type fm32 = makeType<RingTy>(SE_I32, 32);
  EXPECT_EQ(fm32.size(), 4);
  EXPECT_TRUE(fm32.isa<RingTy>());
  EXPECT_EQ(fm32.toString(), "Ring<SE_I32,32>");
  EXPECT_EQ(Type::fromString(fm32.toString()), fm32);

  Type fm128 = makeType<RingTy>(SE_I64, 128);
  EXPECT_EQ(fm128.size(), 16);
  EXPECT_TRUE(fm128.isa<RingTy>());
  EXPECT_EQ(fm128.toString(), "Ring<SE_I64,128>");
  EXPECT_EQ(Type::fromString(fm128.toString()), fm128);
}

}  // namespace spu
