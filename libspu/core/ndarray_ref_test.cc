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

#include "libspu/core/ndarray_ref.h"

#include <cstddef>

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
  EXPECT_TRUE(a.shape().isScalar());
  EXPECT_EQ(a.shape().ndim(), 0);
  EXPECT_EQ(a.offset(), 0);
  EXPECT_EQ(a.elsize(), 0);
}

void verify_array(const NdArrayRef &in, const std::vector<int32_t> &expected) {
  auto numel = in.numel();

  EXPECT_EQ(numel, (int64_t)expected.size());

  int32_t counter = 0;
  for (auto iter = in.cbegin(); iter != in.cend(); ++iter, ++counter) {
    EXPECT_EQ(iter.getScalarValue<std::int32_t>(), expected[counter]);
    // If somehow iter goes into infinite loop, this is a safe guard
    ASSERT_LT(counter, numel);
  }
}

TEST(NdArrayRefTest, Iterator) {
  NdArrayRef a(std::make_shared<yacl::Buffer>(9 * sizeof(int32_t)),
               makePtType(PT_I32), {3, 3}, {3, 1}, 0);

  EXPECT_EQ(a.numel(), 9);

  // Fill array with 0 1 2 3 4 5
  std::iota(a.data<int32_t>(), a.data<int32_t>() + 9, 0);

  verify_array(a, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(NdArrayRefTest, StridedIterator) {
  // Make 3x3 element, strides = 2x2 array
  NdArrayRef a(std::make_shared<yacl::Buffer>(36 * sizeof(int32_t)),
               makePtType(PT_I32), {3, 3}, {2L * 6, 2}, 0);

  EXPECT_EQ(a.numel(), 9);

  // Fill array with 0 1 2 3 4 5
  std::iota(a.data<int32_t>(), a.data<int32_t>() + 36, 0);

  verify_array(a, std::vector<int32_t>{0, 2, 4, 12, 14, 16, 24, 26, 28});
}

TEST(NdArrayRefTest, NdStrides) {
  // Make 3x3 element, strides = 2x2 array
  NdArrayRef a(std::make_shared<yacl::Buffer>(36 * sizeof(int32_t)),
               makePtType(PT_I32), {3, 3}, {2L * 6, 2}, 0);

  EXPECT_EQ(a.numel(), 9);

  // Fill array with 0 1 2 3 4 5
  std::iota(a.data<int32_t>(), a.data<int32_t>() + 36, 0);

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

TEST(NdArrayRefTest, UpdateSlice) {
  // Make 3x3 element, strides = 2x2 array
  NdArrayRef a(std::make_shared<yacl::Buffer>(9 * sizeof(int32_t)),
               makePtType(PT_I32), {3, 3}, {3, 1}, 0);
  NdArrayRef b(std::make_shared<yacl::Buffer>(4 * sizeof(int32_t)),
               makePtType(PT_I32), {2, 2}, {2, 1}, 0);
  // a : 0,1,2
  //     3,4,5
  //     6,7,8
  // b : 0,1
  //     2,3
  std::iota(a.data<int32_t>(), a.data<int32_t>() + 9, 0);
  std::iota(b.data<int32_t>(), b.data<int32_t>() + 4, 0);

  a.update_slice(b, {0, 1});
  // a : 0,0,1
  //     3,2,3
  //     6,7,8
  EXPECT_EQ(a.at<int32_t>({0, 0}), 0);
  EXPECT_EQ(a.at<int32_t>({0, 1}), 0);
  EXPECT_EQ(a.at<int32_t>({0, 2}), 1);
  EXPECT_EQ(a.at<int32_t>({1, 0}), 3);
  EXPECT_EQ(a.at<int32_t>({1, 1}), 2);

  EXPECT_THROW(a.update_slice(b, {0, 2}), ::yacl::EnforceNotMet);
}

TEST(NdArrayRefTest, Indexing) {
  auto buf = std::make_shared<yacl::Buffer>(24 * sizeof(int32_t));
  // Fill array with 0 1 2 3 4 5
  std::iota(static_cast<int32_t *>(buf->data()),
            static_cast<int32_t *>(buf->data()) + 24, 0);
  {
    // 3x2 compact
    NdArrayRef arr(buf, makePtType(PT_I32), {3, 2}, {2, 1}, 0);
    EXPECT_TRUE(arr.canUseFastIndexing());

    verify_array(arr, std::vector<int32_t>{0, 1, 2, 3, 4, 5});
  }
  {
    // 3 x 2 non-compact, but can do fast indexing
    // c x nc
    NdArrayRef arr(buf, makePtType(PT_I32), {3, 2}, {4, 2}, 0);
    EXPECT_TRUE(arr.canUseFastIndexing());

    verify_array(arr, std::vector<int32_t>{0, 2, 4, 6, 8, 10});
  }

  {
    // 3 x 2 non-compact, cannot do fast indexing
    NdArrayRef arr(buf, makePtType(PT_I32), {3, 2}, {8, 2}, 0);
    EXPECT_FALSE(arr.canUseFastIndexing());

    verify_array(arr, std::vector<int32_t>{0, 2, 8, 10, 16, 18});
  }
}

}  // namespace spu
