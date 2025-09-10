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

#include "libspu/mpc/cheetah/ot/matrix_transpose.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yacl/base/aligned_vector.h"

namespace spu::mpc::cheetah::test {

TEST(MatrixTranspose, TypeTraitTest) {
  EXPECT_EQ(is_uint_v<uint32_t>, true);
  EXPECT_EQ(is_uint_v<uint64_t>, true);
  EXPECT_EQ(is_uint_v<uint128_t>, true);

  EXPECT_EQ(is_uint_v<int32_t>, false);
  EXPECT_EQ(is_uint_v<int64_t>, false);
  EXPECT_EQ(is_uint_v<int128_t>, false);

  EXPECT_EQ(is_uint_v<uint8_t>, false);
  EXPECT_EQ(is_uint_v<uint16_t>, false);
}

template <typename T>
class NaiveTransposeTest : public ::testing::Test {};

using ValidType = ::testing::Types<uint32_t, uint64_t, uint128_t>;

TYPED_TEST_SUITE(NaiveTransposeTest, ValidType);

TYPED_TEST(NaiveTransposeTest, NaiveTranspose) {
  using Dt = TypeParam;

  const size_t m = 100;
  const size_t n = 10;
  const auto numel = m * n;

  auto inp = yacl::UninitAlignedVector<Dt>(numel);
  std::generate_n(inp.begin(), numel, [n = 0]() mutable { return n++; });

  // check origin m*n matrix
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_EQ(inp[i * n + j], static_cast<Dt>(i * n + j));
    }
  }

  // transpose
  auto oup = yacl::UninitAlignedVector<Dt>(numel);
  naive_transpose(inp.data(), oup.data(), m, n);

  // check transposed n*m matrix
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      EXPECT_EQ(oup[i * m + j], static_cast<Dt>(j * n + i));
    }
  }
}

TYPED_TEST(NaiveTransposeTest, CacheFriendlyTranspose) {
  using Dt = TypeParam;

  const size_t m = 100;
  const size_t n = 10;
  const auto numel = m * n;

  auto inp = yacl::UninitAlignedVector<Dt>(numel);
  std::generate_n(inp.begin(), numel, [n = 0]() mutable { return n++; });

  // check origin m*n matrix
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_EQ(inp[i * n + j], static_cast<Dt>(i * n + j));
    }
  }

  // transpose
  auto oup = yacl::UninitAlignedVector<Dt>(numel);
  cache_friendly_transpose(inp.data(), oup.data(), m, n);

  // check transposed n*m matrix
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      EXPECT_EQ(oup[i * m + j], static_cast<Dt>(j * n + i));
    }
  }
}

TYPED_TEST(NaiveTransposeTest, SseTransposeTest) {
  using Dt = TypeParam;

  const size_t m = 16;
  const size_t n = 16;
  const auto numel = m * n;
  auto inp = yacl::UninitAlignedVector<Dt>(numel);

  std::generate_n(inp.begin(), numel, [n = 0]() mutable { return n++; });

  // check origin m*n matrix
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_EQ(inp[i * n + j], static_cast<Dt>(i * n + j));
    }
  }

  // transpose
  auto oup = yacl::UninitAlignedVector<Dt>(numel);
  sse_transpose(inp.data(), oup.data(), m, n);

  // check transposed n*m matrix
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      EXPECT_EQ(oup[i * m + j], static_cast<Dt>(j * n + i));
    }
  }
}
}  // namespace spu::mpc::cheetah::test