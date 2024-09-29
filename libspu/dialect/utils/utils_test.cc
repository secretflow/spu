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

#include "libspu/dialect/utils/utils.h"

#include "gtest/gtest.h"

namespace mlir::spu {

TEST(UtilsTest, APIntConversion) {
  // int128_t to APInt
  {
    int128_t in = 8;
    APInt ret = convertFromInt128(32, in);

    EXPECT_EQ(ret.getBitWidth(), 32);
    EXPECT_EQ(ret.getSExtValue(), in);
  }

  {
    int128_t in = -8;
    APInt ret = convertFromInt128(32, in);

    EXPECT_EQ(ret.getBitWidth(), 32);
    EXPECT_EQ(ret.getSExtValue(), in);
  }

  // int128_t to APInt
  {
    int128_t in = static_cast<int128_t>(1) << 70;
    APInt ret = convertFromInt128(128, in);

    EXPECT_EQ(ret.getBitWidth(), 128);
    EXPECT_EQ(ret.getNumWords(), 2);
    EXPECT_EQ(ret.getActiveBits(), 71);
  }

  {
    int128_t in = -(static_cast<int128_t>(1) << 70);
    APInt ret = convertFromInt128(128, in);

    EXPECT_EQ(ret.getNumWords(), 2);
    EXPECT_EQ(ret.getActiveBits(), 128);
  }

  // APInt to int128_t
  {
    APInt in(128, ArrayRef<uint64_t>{static_cast<uint64_t>(-64),
                                     static_cast<uint64_t>(-1)});

    int128_t ret = convertFromAPInt(in);

    EXPECT_EQ(ret, -64);
  }
}

}  // namespace mlir::spu
