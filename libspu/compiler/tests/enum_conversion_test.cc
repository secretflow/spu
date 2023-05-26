// Copyright 2023 Ant Group Co., Ltd.
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

#include "gtest/gtest.h"

#include "libspu/dialect/pphlo_base_enums.h"

#include "libspu/spu.pb.h"

namespace spu {

// Compiler options relies on pb and td has the same string representation of
// fields, use a round trip test to lock this down
TEST(EnumConversion, Public) {
#define CHECK(T)                                                               \
  auto v = Visibility::T;                                                      \
  auto mlir_v =                                                                \
      mlir::pphlo::symbolizeEnum<mlir::pphlo::Visibility>(Visibility_Name(v)); \
  EXPECT_EQ(mlir_v, mlir::pphlo::Visibility::T);

  {CHECK(VIS_PUBLIC)} { CHECK(VIS_SECRET) }

#undef CHECK
}

TEST(EnumConversion, ProtoKinds) {
  Visibility v = Visibility::VIS_PUBLIC;
  // This is basically a compile time check...that there is no new field
  switch (v) {
  case Visibility::VIS_SECRET: // NOLINT
    break;
  case Visibility::VIS_PUBLIC:
    break;
  case Visibility::VIS_INVALID:
    break;
  case Visibility::Visibility_INT_MAX_SENTINEL_DO_NOT_USE_:
    break;
  case Visibility::Visibility_INT_MIN_SENTINEL_DO_NOT_USE_:
    break;
  }
}

TEST(EnumConversion, TdKinds) {
  mlir::pphlo::Visibility v = mlir::pphlo::Visibility::VIS_PUBLIC;
  // This is basically a compile time check...that there is no new field
  switch (v) {
  case mlir::pphlo::Visibility::VIS_SECRET: // NOLINT
    break;
  case mlir::pphlo::Visibility::VIS_PUBLIC:
    break;
  }
}

} // namespace spu
