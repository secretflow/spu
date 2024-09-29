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

#include "libspu/core/xt_helper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(ArrayRefUtilTest, XmemrefSanity) {
  // create a non-empty ndarray
  xt::xarray<double> a = {{0., 1., 2.}, {3., 4., 5.}};
  {
    ASSERT_THAT(a.shape(), testing::ElementsAre(2, 3));
    ASSERT_THAT(a.strides(), testing::ElementsAre(3, 1));
    ASSERT_EQ(a.size(), 6);
  }

  // Assign to scalar, make it a 0-dimension array.
  // see: https://xtensor.readthedocs.io/en/latest/scalar.html
  {
    double s = 1.2;
    a = s;

    ASSERT_TRUE(a.shape().empty());
    ASSERT_TRUE(a.strides().empty());
    ASSERT_EQ(a.size(), 1);
  }

  // create a empty ndarray
  xt::xarray<double> b;
  {
    ASSERT_TRUE(b.shape().empty());
    ASSERT_TRUE(b.strides().empty());
    ASSERT_EQ(b.size(), 1);
  }
}

}  // namespace spu
