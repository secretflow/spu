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

#include "spu/core/ndarray_ref.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(NdArrayRefTest, Empty) {
  // numpy behaviour.
  // >>> x = np.zeros(())
  // >>> x.shape
  // ()
  // >>> x.size
  // 1
  // >>> x.strides
  // ()
  NdArrayRef a;
  EXPECT_EQ(a.numel(), 1);
  EXPECT_TRUE(a.strides().empty());
  EXPECT_TRUE(a.shape().empty());
  EXPECT_EQ(a.offset(), 0);
  EXPECT_EQ(a.elsize(), 0);
}

TEST(ArrayRefTest, NdStrides) {
  // Make 3x3 element, strides = 2x2 array
  NdArrayRef a(std::make_shared<yasl::Buffer>(36 * sizeof(int32_t)),
               makePtType(PT_I32), {3, 3}, {2 * 6, 2}, 0);

  EXPECT_EQ(a.numel(), 9);

  // Fill array with 0 1 2 3 4 5
  std::iota(static_cast<int32_t*>(a.data()),
            static_cast<int32_t*>(a.data()) + 36, 0);

  EXPECT_EQ(a.at<int32_t>({0, 0}), 0);
  EXPECT_EQ(a.at<int32_t>({0, 1}), 2);
  EXPECT_EQ(a.at<int32_t>({1, 1}), 14);
  EXPECT_EQ(a.at<int32_t>({2, 2}), 28);

  // Make a compact clone
  auto b = a.clone();

  EXPECT_TRUE(b.isCompact());
  EXPECT_EQ(b.numel(), 9);
  EXPECT_EQ(b.strides(), std::vector<int64_t>({3, 1}));

  EXPECT_EQ(b.at<int32_t>({0, 0}), 0);
  EXPECT_EQ(b.at<int32_t>({0, 1}), 2);
  EXPECT_EQ(b.at<int32_t>({1, 1}), 14);
  EXPECT_EQ(b.at<int32_t>({2, 2}), 28);
}

}  // namespace spu
