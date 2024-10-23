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
  EXPECT_FALSE(va.isa<PtTy>());
  EXPECT_TRUE(va.isa<VoidTy>());
  EXPECT_EQ(va.size(), 0);
  EXPECT_EQ(va.toString(), "Void<>");
  EXPECT_EQ(Type::fromString(va.toString()), va);

  Type vb = va;  // NOLINT: Test copy ctor
  EXPECT_FALSE(vb.isa<PtTy>());
  EXPECT_EQ(vb.size(), 0);
  EXPECT_EQ(vb.toString(), "Void<>");

  EXPECT_EQ(va, vb);
  EXPECT_EQ(vb, va);
}

TEST(TypeTest, PtTy) {
  // extern Type Void;
  Type I8 = makePtType(PT_I8);
  EXPECT_EQ(I8.size(), 1);
  EXPECT_TRUE(I8.isa<PtTy>());
  EXPECT_EQ(I8.toString(), "Plaintext<PT_I8>");
  EXPECT_EQ(Type::fromString(I8.toString()), I8);

  Type U8 = makePtType(PT_U8);
  EXPECT_EQ(U8.size(), 1);
  EXPECT_TRUE(U8.isa<PtTy>());
  EXPECT_EQ(U8.toString(), "Plaintext<PT_U8>");
  EXPECT_EQ(Type::fromString(U8.toString()), U8);

  Type I16 = makePtType(PT_I16);
  EXPECT_EQ(I16.size(), 2);
  EXPECT_TRUE(I16.isa<PtTy>());
  EXPECT_EQ(I16.toString(), "Plaintext<PT_I16>");
  EXPECT_EQ(Type::fromString(I16.toString()), I16);

  Type U16 = makePtType(PT_U16);
  EXPECT_EQ(U16.size(), 2);
  EXPECT_TRUE(U16.isa<PtTy>());
  EXPECT_EQ(U16.toString(), "Plaintext<PT_U16>");
  EXPECT_EQ(Type::fromString(U16.toString()), U16);

  Type I32 = makePtType(PT_I32);
  EXPECT_EQ(I32.size(), 4);
  EXPECT_TRUE(I32.isa<PtTy>());
  EXPECT_EQ(I32.toString(), "Plaintext<PT_I32>");
  EXPECT_EQ(Type::fromString(I32.toString()), I32);

  Type U32 = makePtType(PT_U32);
  EXPECT_EQ(U32.size(), 4);
  EXPECT_TRUE(U32.isa<PtTy>());
  EXPECT_EQ(U32.toString(), "Plaintext<PT_U32>");
  EXPECT_EQ(Type::fromString(U32.toString()), U32);

  Type I64 = makePtType(PT_I64);
  EXPECT_EQ(I64.size(), 8);
  EXPECT_TRUE(I64.isa<PtTy>());
  EXPECT_EQ(I64.toString(), "Plaintext<PT_I64>");
  EXPECT_EQ(Type::fromString(I64.toString()), I64);

  Type U64 = makePtType(PT_U64);
  EXPECT_EQ(U64.size(), 8);
  EXPECT_TRUE(U64.isa<PtTy>());
  EXPECT_EQ(U64.toString(), "Plaintext<PT_U64>");
  EXPECT_EQ(Type::fromString(U64.toString()), U64);

  Type F32 = makePtType(PT_F32);
  EXPECT_EQ(F32.size(), 4);
  EXPECT_TRUE(F32.isa<PtTy>());
  EXPECT_EQ(F32.toString(), "Plaintext<PT_F32>");
  EXPECT_EQ(Type::fromString(F32.toString()), F32);

  Type F64 = makePtType(PT_F64);
  EXPECT_EQ(F64.size(), 8);
  EXPECT_TRUE(F64.isa<PtTy>());
  EXPECT_EQ(F64.toString(), "Plaintext<PT_F64>");
  EXPECT_EQ(Type::fromString(F64.toString()), F64);

  Type I128 = makePtType(PT_I128);
  EXPECT_EQ(I128.size(), 16);
  EXPECT_TRUE(I128.isa<PtTy>());
  EXPECT_EQ(I128.toString(), "Plaintext<PT_I128>");
  EXPECT_EQ(Type::fromString(I128.toString()), I128);

  Type U128 = makePtType(PT_U128);
  EXPECT_EQ(U128.size(), 16);
  EXPECT_TRUE(U128.isa<PtTy>());
  EXPECT_EQ(U128.toString(), "Plaintext<PT_U128>");
  EXPECT_EQ(Type::fromString(U128.toString()), U128);
}

TEST(TypeTest, RingTy) {
  Type fm32 = makeType<RingTy>(FM32);
  EXPECT_EQ(fm32.size(), 4);
  EXPECT_TRUE(fm32.isa<RingTy>());
  EXPECT_EQ(fm32.toString(), "Ring<FM32>");
  EXPECT_EQ(Type::fromString(fm32.toString()), fm32);

  Type fm128 = makeType<RingTy>(FM128);
  EXPECT_EQ(fm128.size(), 16);
  EXPECT_TRUE(fm128.isa<RingTy>());
  EXPECT_EQ(fm128.toString(), "Ring<FM128>");
  EXPECT_EQ(Type::fromString(fm128.toString()), fm128);
}

TEST(TypeTest, GfmpTy) {
  Type gfmp31 = makeType<GfmpTy>(FM32);
  EXPECT_EQ(gfmp31.size(), 4);
  EXPECT_TRUE(gfmp31.isa<GfmpTy>());
  EXPECT_EQ(gfmp31.toString(), "Gfmp<FM32,31>");
  EXPECT_EQ(Type::fromString(gfmp31.toString()), gfmp31);

  Type gfmp61 = makeType<GfmpTy>(FM64);
  EXPECT_EQ(gfmp61.size(), 8);
  EXPECT_TRUE(gfmp61.isa<GfmpTy>());
  EXPECT_EQ(gfmp61.toString(), "Gfmp<FM64,61>");
  EXPECT_EQ(Type::fromString(gfmp61.toString()), gfmp61);

  Type gfmp127 = makeType<GfmpTy>(FM128);
  EXPECT_EQ(gfmp127.size(), 16);
  EXPECT_TRUE(gfmp127.isa<GfmpTy>());
  EXPECT_EQ(gfmp127.toString(), "Gfmp<FM128,127>");
  EXPECT_EQ(Type::fromString(gfmp127.toString()), gfmp127);
}

}  // namespace spu
