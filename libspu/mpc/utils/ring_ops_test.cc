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

#include "libspu/mpc/utils/ring_ops.h"

#include <random>

#include "gtest/gtest.h"

namespace spu::mpc {

class RingArrayRefTest
    : public ::testing::TestWithParam<  //
          std::tuple<FieldType,
                     int64_t,  // numel
                     int64_t,  // stride of first param, if have
                     int64_t   // stride of second param, if have
                     >> {};

static ArrayRef makeRandomArray(FieldType field, size_t numel, size_t stride) {
  const Type ty = makeType<RingTy>(field);
  const size_t buf_size = SizeOf(field) * numel * stride;
  // make random buffer.
  auto buf = std::make_shared<yacl::Buffer>(buf_size);
  {
    size_t numOfInts = buf_size / sizeof(int32_t);
    auto* begin = buf->data<int32_t>();
    for (size_t idx = 0; idx < numOfInts; idx++) {
      *(begin + idx) = std::rand();
    }
  }

  const int64_t offset = 0;
  return ArrayRef(buf, ty, numel, stride, offset);
}

INSTANTIATE_TEST_SUITE_P(
    RingArrayRefTestSuite, RingArrayRefTest,
    testing::Combine(testing::Values(FM32, FM64, FM128),  //
                     testing::Values(1, 3, 1000),         // size of parameters
                     testing::Values(1, 3),  // stride of first param
                     testing::Values(1, 3)   // stride of second param
                     ),
    [](const testing::TestParamInfo<RingArrayRefTest::ParamType>& p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

TEST_P(RingArrayRefTest, Assign) {
  const FieldType field = std::get<0>(GetParam());
  const int64_t numel = std::get<1>(GetParam());
  const int64_t stride = std::get<2>(GetParam());
  const Type ty = makeType<RingTy>(field);

  // GIVEN
  const ArrayRef y = makeRandomArray(field, numel, stride);
  ArrayRef x(ty, numel);

  // WHEN
  ring_assign(x, y);  // x = y;

  // THEN
  EXPECT_TRUE(ring_all_equal(x, y));
}

TEST_P(RingArrayRefTest, Negate) {
  const FieldType field = std::get<0>(GetParam());
  const int64_t numel = std::get<1>(GetParam());
  const int64_t stride = std::get<2>(GetParam());
  const Type ty = makeType<RingTy>(field);

  {
    // GIVEN
    const ArrayRef y = makeRandomArray(field, numel, stride);

    // WHEN
    auto x = ring_neg(y);

    // THEN
    EXPECT_FALSE(ring_all_equal(x, y));
    EXPECT_TRUE(ring_all_equal(ring_neg(x), y));
    EXPECT_TRUE(ring_all_equal(x, ring_neg(y)));
  }

  {
    // GIVEN
    ArrayRef z = makeRandomArray(field, numel, stride);
    auto zbuf = z.buf();
    const ArrayRef w = z.clone();

    // THEN
    ring_neg_(z);
    EXPECT_EQ(z.buf(), zbuf);
    EXPECT_FALSE(ring_all_equal(z, w));

    ring_neg_(z);
    EXPECT_EQ(z.buf(), zbuf);
    EXPECT_TRUE(ring_all_equal(z, w));
  }
}

TEST_P(RingArrayRefTest, Not) {
  const FieldType field = std::get<0>(GetParam());
  const int64_t numel = std::get<1>(GetParam());
  const int64_t stride = std::get<2>(GetParam());
  const Type ty = makeType<RingTy>(field);

  {
    // GIVEN
    const ArrayRef y = makeRandomArray(field, numel, stride);

    // WHEN
    auto x = ring_not(y);

    // THEN
    EXPECT_FALSE(ring_all_equal(x, y));
    EXPECT_TRUE(ring_all_equal(ring_not(x), y));
    EXPECT_TRUE(ring_all_equal(x, ring_not(y)));
  }

  {
    // GIVEN
    ArrayRef z = makeRandomArray(field, numel, stride);
    auto zbuf = z.buf();
    const ArrayRef w = z.clone();

    // THEN
    ring_not_(z);
    EXPECT_EQ(z.buf(), zbuf);
    EXPECT_FALSE(ring_all_equal(z, w));

    ring_not_(z);
    EXPECT_EQ(z.buf(), zbuf);
    EXPECT_TRUE(ring_all_equal(z, w));
  }
}

TEST_P(RingArrayRefTest, ReverseBit) {
  const FieldType field = std::get<0>(GetParam());
  const int64_t numel = std::get<1>(GetParam());
  const int64_t stride = std::get<2>(GetParam());
  const Type ty = makeType<RingTy>(field);

  {
    // GIVEN
    const ArrayRef y = makeRandomArray(field, numel, stride);

    // WHEN
    auto x = ring_bitrev(y, 10, 20);

    // THEN
    EXPECT_FALSE(ring_all_equal(x, y));
    EXPECT_TRUE(ring_all_equal(ring_bitrev(x, 10, 20), y));
  }

  {
    // GIVEN
    ArrayRef z = makeRandomArray(field, numel, stride);
    auto zbuf = z.buf();
    const ArrayRef w = z.clone();

    // THEN
    ring_bitrev_(z, 10, 20);
    EXPECT_EQ(z.buf(), zbuf);
    EXPECT_FALSE(ring_all_equal(z, w));

    ring_bitrev_(z, 10, 20);
    EXPECT_EQ(z.buf(), zbuf);
    EXPECT_TRUE(ring_all_equal(z, w));
  }
}

TEST_P(RingArrayRefTest, RingBitMask) {
  const FieldType field = std::get<0>(GetParam());
  const int64_t numel = std::get<1>(GetParam());
  const int64_t stride = std::get<2>(GetParam());
  const Type ty = makeType<RingTy>(field);

  {
    // GIVEN
    const ArrayRef x = makeRandomArray(field, numel, stride);
    const size_t step = SizeOf(field) * 8 / 4;

    // WHEN
    auto y0 = ring_bitmask(x, step * 0, step * 1);
    auto y1 = ring_bitmask(x, step * 1, step * 2);
    auto y2 = ring_bitmask(x, step * 2, step * 3);
    auto y3 = ring_bitmask(x, step * 3, step * 4);

    // and to zero
    auto z = ring_and(y0, y1);
    ring_and_(z, y2);
    ring_and_(z, y3);

    // sum to original
    auto s = ring_sum({y0, y1, y2, y3});

    // THEN
    EXPECT_TRUE(ring_all_equal(s, x));
    EXPECT_TRUE(ring_all_equal(z, ring_zeros(field, numel)));
  }
  {
    // GIVEN
    const ArrayRef x = makeRandomArray(field, numel, stride);
    const size_t step = SizeOf(field) * 8 / 4;

    // WHEN
    auto y0 = x.clone();
    auto y1 = x.clone();
    auto y2 = x.clone();
    auto y3 = x.clone();
    ring_bitmask_(y0, step * 0, step * 1);
    ring_bitmask_(y1, step * 1, step * 2);
    ring_bitmask_(y2, step * 2, step * 3);
    ring_bitmask_(y3, step * 3, step * 4);

    auto z = ring_and(y0, y1);
    ring_and_(z, y2);
    ring_and_(z, y3);

    auto s = ring_sum({y0, y1, y2, y3});

    // THEN
    EXPECT_TRUE(ring_all_equal(s, x));
    EXPECT_TRUE(ring_all_equal(z, ring_zeros(field, numel)));
  }
}

}  // namespace spu::mpc
