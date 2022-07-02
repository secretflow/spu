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

#include "spu/core/xt_helper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(ArrayRefUtilTest, XtensorSanity) {
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

TEST(ArrayRefUtilTest, MakeNdArray) {
  // make from scalar
  {
    double _ = 1.0f;
    auto a = xt_to_ndarray(_);
    EXPECT_EQ(a.eltype(), F64);
    EXPECT_EQ(a.buf()->size(), a.elsize());
    EXPECT_EQ(a.numel(), 1);
    EXPECT_TRUE(a.shape().empty());
    EXPECT_TRUE(a.strides().empty());
    EXPECT_EQ(a.offset(), 0);
    EXPECT_EQ(a.at<double>({}), 1.0f);

    EXPECT_THROW(xt_adapt<float>(a), std::exception);
  }

  // make from xcontainer.
  {
    xt::xarray<double> _ = {{0., 1., 2.}, {3., 4., 5.}};
    auto a = xt_to_ndarray(_);
    EXPECT_EQ(a.buf()->size(), a.elsize() * _.size());
    EXPECT_EQ(a.eltype(), F64);
    EXPECT_EQ(a.numel(), _.size());
    EXPECT_EQ(a.shape().size(), _.shape().size());
    for (size_t idx = 0; idx < a.shape().size(); ++idx) {
      EXPECT_EQ(a.shape()[idx], _.shape()[idx]);
    }
    EXPECT_EQ(a.strides(),
              std::vector<int64_t>(_.strides().begin(), _.strides().end()));
    EXPECT_EQ(a.offset(), 0);
    EXPECT_EQ(xt_adapt<double>(a), _);

    EXPECT_THROW(xt_adapt<float>(a), std::exception);
  }

  // make from xexpression
  {
    xt::xarray<double> _ = {{0., 1., 2.}, {3., 4., 5.}};
    auto a = xt_to_ndarray(_ * _ + _);
    EXPECT_EQ(a.buf()->size(), a.elsize() * _.size());
    EXPECT_EQ(a.eltype(), F64);
    EXPECT_EQ(a.numel(), _.size());
    EXPECT_EQ(a.shape().size(), _.shape().size());
    for (size_t idx = 0; idx < a.shape().size(); ++idx) {
      EXPECT_EQ(a.shape()[idx], _.shape()[idx]);
    }
    EXPECT_EQ(a.strides(),
              std::vector<int64_t>(_.strides().begin(), _.strides().end()));
    EXPECT_EQ(a.offset(), 0);
    EXPECT_EQ(xt_adapt<double>(a), _ * _ + _);

    EXPECT_THROW(xt_adapt<float>(a), std::exception);
  }
}

}  // namespace spu
