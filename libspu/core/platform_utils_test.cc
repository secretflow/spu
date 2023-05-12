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

#include "libspu/core/platform_utils.h"

#include <array>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(PlatformUtils, pdep_u64) {
  auto x = pdep_u64(19, 19);

  GTEST_ASSERT_GT(x, 0);
}

TEST(PlatformUtils, pext_u64) {
  auto x = pext_u64(19, 19);

  GTEST_ASSERT_GT(x, 0);
}

}  // namespace spu
