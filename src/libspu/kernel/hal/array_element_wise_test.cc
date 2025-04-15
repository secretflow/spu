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

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {

TEST(ArrayElementwiseTest, NegConstantF32) {
  xt::xarray<float> in = {-2.5F, 3.14F, 2.25F, -10.0F, 6.0F};

  auto ret = test::evalUnaryOp<float>(VIS_PUBLIC, negate, in);

  xt::xarray<float> expected = {2.5F, -3.14F, -2.25F, 10.0F, -6.0F};

  EXPECT_TRUE(xt::allclose(ret, expected));
}

TEST(ArrayElementwiseTest, NegConstantF64) {
  xt::xarray<double> in = {-2.5, 3.14, 2.25, -10.0, 6.0};

  auto ret = test::evalUnaryOp<float>(VIS_PUBLIC, negate, in);

  xt::xarray<double> expected = {2.5, -3.14, -2.25, 10.0, -6.0};

  EXPECT_TRUE(xt::allclose(ret, expected));
}

TEST(ArrayElementwiseTest, NegConstantS32) {
  xt::xarray<int32_t> in = {-1,
                            0,
                            1,
                            324,
                            std::numeric_limits<int32_t>::min(),
                            std::numeric_limits<int32_t>::max()};

  auto ret = test::evalUnaryOp<int32_t>(VIS_PUBLIC, negate, in);

  // -min == max for int32_t due to an overflow. In C++ it is undefined
  // behavior to do this calculation.
  xt::xarray<int32_t> expected = {
      1,
      0,  // fixed-point
      -1,
      -324,
      std::numeric_limits<int32_t>::min(),  // fixed-point
      -std::numeric_limits<int32_t>::max()};
  EXPECT_EQ(ret, expected);
}

TEST(ArrayElementwiseTest, NegConstantS64) {
  xt::xarray<int64_t> in = {
      -1,
      1,
      0,
      0x12345678,
      static_cast<int64_t>(0xffffffff12345678L),
      static_cast<int64_t>(0x8000000000000000LL),
      static_cast<int64_t>(0x8000000000000001LL),
  };

  auto ret = test::evalUnaryOp<float>(VIS_PUBLIC, negate, in);

  xt::xarray<int64_t> expected = {
      1,
      -1,
      0,
      -0x12345678,
      0xedcba988,
      static_cast<int64_t>(0x8000000000000000LL),
      -static_cast<int64_t>(0x8000000000000001LL),
  };
  EXPECT_EQ(ret, expected);
}

TEST(ArrayElementwiseTest, IntPow) {
  xt::xarray<int32_t> lhs = {0, 1, 2, 3, 4, 5, -1, -2, 3, 5, 3, 1};
  xt::xarray<int32_t> rhs = {0, 3, 3, 3, 3, 3, 2, 3, 2, 10, -100, -2};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, power, lhs, rhs);

  xt::xarray<int32_t> expected = {1, 1,  8, 27,      64, 125,
                                  1, -8, 9, 9765625, 0,  1};

  // FIXME: figure out why it's wrong
  // EXPECT_EQ(ret, expected);
}

TEST(ArrayElementwiseTest, IntPowLarge) {
  xt::xarray<int64_t> lhs = {2};
  xt::xarray<int64_t> rhs = {62};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, power, lhs, rhs);

  xt::xarray<int64_t> expected = {4611686018427387904};

  // FIXME: figure out why it's wrong
  // EXPECT_EQ(ret, expected);
}

TEST(ArrayElementwiseTest, AddTwoConstantF32s) {
  xt::xarray<float> lhs = {-2.5F, 3.14F, 2.25F, -10.0F, 6.0F};
  xt::xarray<float> rhs = {100.0F, 3.13F, 2.75F, 10.5F, -999.0F};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, add, lhs, rhs);

  xt::xarray<float> expected = {97.5F, 6.27F, 5.0F, 0.5F, -993.0F};

  EXPECT_TRUE(xt::allclose(ret, expected, 0.001, 0.0001)) << ret;
}

TEST(ArrayElementwiseTest, AddTwoConstantU64s) {
  xt::xarray<uint64_t> lhs = {0xFFFFFFFF,
                              static_cast<uint64_t>(-1),
                              0,
                              0,
                              0x7FFFFFFFFFFFFFFFLL,
                              0x7FFFFFFFFFFFFFFLL,
                              0x8000000000000000ULL,
                              0x8000000000000000ULL,
                              1};
  xt::xarray<uint64_t> rhs = {1,
                              0x7FFFFFFFFFFFFFFLL,
                              0x7FFFFFFFFFFFFFFFLL,
                              0x8000000000000000ULL,
                              0,
                              static_cast<uint64_t>(-1),
                              0,
                              1,
                              0x8000000000000000ULL};

  auto ret =
      test::evalBinaryOp<uint64_t>(VIS_PUBLIC, VIS_PUBLIC, add, lhs, rhs);

  xt::xarray<uint64_t> expected = lhs + rhs;

  EXPECT_EQ(ret, expected);
}

TEST(ArrayElementwiseTest, SubTwoConstantS64s) {
  xt::xarray<int64_t> lhs = {static_cast<int64_t>(0x8000000000000000LL),
                             static_cast<int64_t>(0x8000000000000000LL),
                             -1,
                             0x7FFFFFFFFFFFFFFLL,
                             0x7FFFFFFFFFFFFFFFLL,
                             1,
                             0,
                             -1};
  xt::xarray<int64_t> rhs = {-1,
                             0,
                             static_cast<int64_t>(0x8000000000000000LL),
                             1,
                             0,
                             0x7FFFFFFFFFFFFFFLL,
                             0x7FFFFFFFFFFFFFFFLL,
                             0x7FFFFFFFFFFFFFFFLL};

  auto ret = test::evalBinaryOp<int64_t>(VIS_PUBLIC, VIS_PUBLIC, sub, lhs, rhs);

  xt::xarray<int64_t> expected = lhs - rhs;

  EXPECT_EQ(ret, expected);
}

TEST(ArrayElementwiseTest, SubTwoConstantF32s) {
  xt::xarray<float> lhs = {-2.5F, 3.14F, 2.25F, -10.0F, 6.0F};
  xt::xarray<float> rhs = {100.0F, 3.13F, 2.75F, 10.5F, -999.0F};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, sub, lhs, rhs);

  xt::xarray<float> expected = {-102.5F, 0.01F, -0.5F, -20.5F, 1005.0F};

  EXPECT_TRUE(xt::allclose(ret, expected, 0.001, 0.0001)) << ret;
}

TEST(ArrayElementwiseTest, SubTwoConstantS32s) {
  xt::xarray<int32_t> lhs = {-1, 0, 2, 1000000000};
  xt::xarray<int32_t> rhs = {-1, 2, 1, -1};

  auto ret = test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, sub, lhs, rhs);

  xt::xarray<int32_t> expected = {0, -2, 1, 1000000001};

  EXPECT_EQ(ret, expected);
}

TEST(ArrayElementwiseTest, SubTwoConstantF64s) {
  xt::xarray<double> lhs = {-2.5, 3.14, 2.25, -10.0, 6.0};
  xt::xarray<double> rhs = {100.0, 3.13, 2.75, 10.5, -999.0};

  auto ret = test::evalBinaryOp<double>(VIS_PUBLIC, VIS_PUBLIC, sub, lhs, rhs);

  xt::xarray<double> expected = {-102.5, 0.01, -0.5, -20.5, 1005.0};

  EXPECT_TRUE(xt::allclose(ret, expected, 0.001, 0.0001)) << ret;
}

TEST(ArrayElementwiseTest, DivTwoConstantF32s) {
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, -10.0F, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.1F, 1.0F, 10.0F, -6.0F};

  auto ret = test::evalBinaryOp<double>(VIS_PUBLIC, VIS_PUBLIC, div, lhs, rhs);

  xt::xarray<double> expected = {-0.25F, 5.0F, 2.25F, -1.0F, -1.0F};

  EXPECT_TRUE(xt::allclose(ret, expected, 0.001, 0.0001)) << ret;
}

TEST(ArrayElementwiseTest, DivS32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<int32_t> vals = {
    INT32_MIN, INT32_MIN + 1, INT32_MIN + 2, -0x40000000, -0x3ffffffF,
    -271181, -1309, -17, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 17, 26, 101,
    7919, 0x40000000, INT32_MAX - 2, INT32_MAX - 1, INT32_MAX};
  // clang-format on

  std::vector<int32_t> dividends;
  std::vector<int32_t> divisors;
  std::vector<int32_t> quotients;
  for (int32_t divisor : vals) {
    if (divisor != 0) {
      for (int32_t dividend : vals) {
        // Avoid integer overflow.
        if (dividend != INT32_MIN || divisor != -1) {
          dividends.push_back(dividend);
          divisors.push_back(divisor);
          quotients.push_back(dividend / divisor);
        }
      }
    }
  }

  auto ret = test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, div, dividends,
                                         divisors);
  // FIXME: Figure out numeric difference
  // EXPECT_TRUE(xt::allclose(xt::adapt(quotients), ret))
  //     << ret << xt::adapt(quotients);
}

TEST(ArrayElementwiseTest, DivU32s) {
  // clang-format off
  // Some interesting values to test.
  std::vector<uint32_t> vals = {
    0, 1, 2, 17, 101, 3333, 0x7FFFFFFF, 0xABCDEF12, 0xCAFEBEEF, 0x80000000,
    0x80000001, UINT32_MAX - 2, UINT32_MAX - 1, UINT32_MAX};
  // clang-format on

  std::vector<uint32_t> dividends;
  std::vector<uint32_t> divisors;
  std::vector<uint32_t> quotients;
  for (uint32_t divisor : vals) {
    if (divisor != 0) {
      for (uint32_t dividend : vals) {
        dividends.push_back(dividend);
        divisors.push_back(divisor);
        quotients.push_back(dividend / divisor);
      }
    }
  }

  auto ret = test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, div,
                                          dividends, divisors);

  // FIXME: Figure out numeric difference
  // EXPECT_TRUE(xt::allclose(xt::adapt(quotients), ret))
  //     << ret << xt::adapt(quotients);
}

TEST(ArrayElementwiseTest, MulTwoConstantF32s) {
  xt::xarray<float> a = {-2.5F, 25.5F, 2.25F, -10.0F, 6.0F};
  xt::xarray<float> b = {10.0F, 5.0F, 1.0F, 10.0F, -6.0F};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, mul, a, b);

  xt::xarray<float> expected = {-25.0F, 127.5F, 2.25F, -100.0F, -36.0F};

  EXPECT_TRUE(xt::allclose(expected, ret)) << ret << expected;
}

TEST(ArrayElementwiseTest, MulTwoConstantS32s) {
  std::vector<int32_t> data = {0,
                               1,
                               -1,
                               1234,
                               0x1a243514,
                               std::numeric_limits<int32_t>::max(),
                               std::numeric_limits<int32_t>::min()};
  // Form the test data set using all products of 'data' with itself.
  std::vector<int32_t> a_data;
  std::vector<int32_t> b_data;
  std::vector<int32_t> expected;
  for (int32_t a : data) {
    for (int32_t b : data) {
      a_data.push_back(a);
      b_data.push_back(b);
      expected.push_back(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
    }
  }

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, mul, a_data, b_data);

  EXPECT_TRUE(xt::allclose(xt::adapt(expected), ret))
      << ret << xt::adapt(expected);
}

TEST(ArrayElementwiseTest, MulTwoConstantU32s) {
  std::vector<uint32_t> data = {0,          1,          0xDEADBEEF, 1234,
                                0x1a243514, 0xFFFFFFFF, 0x80808080};

  // Form the test data set using all products of 'data' with itself.
  std::vector<uint32_t> a_data;
  std::vector<uint32_t> b_data;
  std::vector<uint32_t> expected;
  for (uint32_t a : data) {
    for (uint32_t b : data) {
      a_data.push_back(a);
      b_data.push_back(b);
      expected.push_back(a * b);
    }
  }

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, mul, a_data, b_data);

  EXPECT_TRUE(xt::allclose(xt::adapt(expected), ret))
      << ret << xt::adapt(expected);
}

TEST(ArrayElementwiseTest, AndPredR1) {
  xt::xarray<bool> a = {false, false, true, true};
  xt::xarray<bool> b = {false, true, false, true};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_and, a, b);

  xt::xarray<bool> expected = {false, false, false, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, AndPredR2) {
  xt::xarray<bool> a = {{false, false}, {true, true}};
  xt::xarray<bool> b = {{false, true}, {false, true}};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_and, a, b);

  xt::xarray<bool> expected = {{false, false}, {false, true}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, AndS32R1) {
  xt::xarray<int32_t> a = {0, -1, -8};
  xt::xarray<int32_t> b = {5, -7, 12};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_and, a, b);

  xt::xarray<int32_t> expected = {0, -7, 8};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, AndS32R2) {
  xt::xarray<int32_t> a = {{0, -5}, {-1, 5}};
  xt::xarray<int32_t> b = {{1, -6}, {4, 5}};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_and, a, b);

  xt::xarray<int32_t> expected = {{0, -6}, {4, 5}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, AndU32R1) {
  xt::xarray<uint32_t> a = {0, 1, 8};
  xt::xarray<uint32_t> b = {5, 7, 12};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_and, a, b);

  xt::xarray<uint32_t> expected = {0, 1, 8};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, AndU32R2) {
  xt::xarray<uint32_t> a = {{0, 1}, {3, 8}};
  xt::xarray<uint32_t> b = {{1, 0}, {7, 6}};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_and, a, b);

  xt::xarray<uint32_t> expected = {{0, 0}, {3, 0}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, OrPredR1) {
  xt::xarray<bool> a = {false, false, true, true};
  xt::xarray<bool> b = {false, true, false, true};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_or, a, b);

  xt::xarray<bool> expected = {false, true, true, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, OrPredR2) {
  xt::xarray<bool> a = {{false, false}, {true, true}};
  xt::xarray<bool> b = {{false, true}, {false, true}};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_or, a, b);

  xt::xarray<bool> expected = {{false, true}, {true, true}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, OrS32R1) {
  xt::xarray<int32_t> a = {0, -1, 8};
  xt::xarray<int32_t> b = {5, -7, 4};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_or, a, b);

  xt::xarray<int32_t> expected = {5, -1, 12};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, OrS32R2) {
  xt::xarray<int32_t> a = {{0, -1}, {8, 8}};
  xt::xarray<int32_t> b = {{5, -7}, {4, 1}};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_or, a, b);

  xt::xarray<int32_t> expected = {{5, -1}, {12, 9}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, OrU32R1) {
  xt::xarray<uint32_t> a = {0, 1, 8};
  xt::xarray<uint32_t> b = {5, 7, 4};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_or, a, b);

  xt::xarray<uint32_t> expected = {5, 7, 12};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, OrU32R2) {
  xt::xarray<uint32_t> a = {{0, 1}, {8, 8}};
  xt::xarray<uint32_t> b = {{5, 7}, {4, 1}};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_or, a, b);

  xt::xarray<uint32_t> expected = {{5, 7}, {12, 9}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, XorPredR1) {
  xt::xarray<bool> a = {false, false, true, true};
  xt::xarray<bool> b = {false, true, false, true};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_xor, a, b);

  xt::xarray<bool> expected = {false, true, true, false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, XorPredR2) {
  xt::xarray<bool> a = {{false, false}, {true, true}};
  xt::xarray<bool> b = {{false, true}, {false, true}};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_xor, a, b);

  xt::xarray<bool> expected = {{false, true}, {true, false}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, XorS32R1) {
  xt::xarray<int32_t> a = {0, -1, 8};
  xt::xarray<int32_t> b = {5, -7, 4};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_xor, a, b);

  xt::xarray<int32_t> expected = {5, 6, 12};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, XorS32R2) {
  xt::xarray<int32_t> a = {{0, -1}, {8, 8}};
  xt::xarray<int32_t> b = {{5, -7}, {4, 1}};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_xor, a, b);

  xt::xarray<int32_t> expected = {{5, 6}, {12, 9}};

  EXPECT_EQ(expected, ret) << ret << expected;
}

TEST(ArrayElementwiseTest, XorU32R1) {
  xt::xarray<uint32_t> a = {0, 1, 8};
  xt::xarray<uint32_t> b = {5, 7, 4};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_xor, a, b);

  xt::xarray<int32_t> expected = {5, 6, 12};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, XorU32R2) {
  xt::xarray<uint32_t> a = {{0, 1}, {8, 8}};
  xt::xarray<uint32_t> b = {{5, 7}, {4, 1}};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, bitwise_xor, a, b);

  xt::xarray<int32_t> expected = {{5, 6}, {12, 9}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, NotPredR1) {
  xt::xarray<bool> a = {false, true, true, false};

  auto ret = test::evalUnaryOp<bool>(VIS_PUBLIC, logical_not, a);

  xt::xarray<bool> expected = {true, false, false, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, NotPredR2) {
  xt::xarray<bool> a = {{false, true}, {true, false}};

  auto ret = test::evalUnaryOp<bool>(VIS_PUBLIC, logical_not, a);

  xt::xarray<bool> expected = {{true, false}, {false, true}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, NotS32R1) {
  xt::xarray<int32_t> a = {-1, 0, 1};

  auto ret = test::evalUnaryOp<int32_t>(VIS_PUBLIC, bitwise_not, a);

  xt::xarray<int32_t> expected = {0, -1, -2};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, NotS32R2) {
  xt::xarray<int32_t> a = {{-1, 0}, {1, 8}};

  auto ret = test::evalUnaryOp<int32_t>(VIS_PUBLIC, bitwise_not, a);

  xt::xarray<int32_t> expected = {{0, -1}, {-2, -9}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, NotU32R1) {
  xt::xarray<uint32_t> a = {0, 4294967295};

  auto ret = test::evalUnaryOp<uint32_t>(VIS_PUBLIC, bitwise_not, a);

  xt::xarray<uint32_t> expected = {4294967295, 0};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, NotU32R2) {
  xt::xarray<uint32_t> a = {{0, 4294967295}, {1, 4294967294}};

  auto ret = test::evalUnaryOp<uint32_t>(VIS_PUBLIC, bitwise_not, a);

  xt::xarray<uint32_t> expected = {{4294967295, 0}, {4294967294, 1}};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareEqF32s) {
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.0F, 2.25F, 10.0F, NAN};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, equal, lhs, rhs);

  xt::xarray<bool> expected = {false, false, true, false, false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareEqF32sTO) {
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.0F, 2.25F, NAN, NAN};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, equal, lhs, rhs);

  xt::xarray<bool> expected = {false, false, true, true, false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareGeF32s) {
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.0F, 1.0F, 10.0F, NAN};

  auto ret = test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, greater_equal,
                                         lhs, rhs);

  xt::xarray<bool> expected = {false, true, true, false, false};

  // FIXME: nan?
  // EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareGeF32sTO) {
  // For portability, need to represent NAN using the following call.
  // The C++ standard does not specify if quiet_NaN() sets the sign bit of
  // its result. The call to std::fabs will ensure that it is not set.
  auto nan = std::fabs(std::numeric_limits<float>::quiet_NaN());
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, nan, 6.0F, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.0F, 1.0F, 10.0F, nan, -nan};

  auto ret = test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, greater_equal,
                                         lhs, rhs);

  xt::xarray<bool> expected = {false, true, true, true, false, true};

  // FIXME: nan?
  // EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareGtF32s) {
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.0F, 1.0F, 10.0F, NAN};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, greater, lhs, rhs);

  xt::xarray<bool> expected = {false, true, true, false, false};

  // FIXME: nan?
  // EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareLeF32s) {
  xt::xarray<float> lhs = {-2.5F, 5.0F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.0F, 1.0F, 10.0F, NAN};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, less_equal, lhs, rhs);

  xt::xarray<bool> expected = {true, true, false, false, false};

  // FIXME: nan?
  // EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareLtF32s) {
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {10.0F, 5.0F, 1.0F, 10.0F, NAN};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, less, lhs, rhs);

  xt::xarray<bool> expected = {true, false, false, false, false};

  // FIXME: nan?
  // EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareEqS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> lhs = {min, min, min, 0, 0, 0, max, max, max};
  xt::xarray<int32_t> rhs = {min, 0, max, -1, 0, 1, min, 0, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, equal, lhs, rhs);

  xt::xarray<bool> expected = {true,  false, false, false, true,
                               false, false, false, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareNeF32s) {
  xt::xarray<float> lhs = {-2.5F, 25.5F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {10.0F, 25.5F, 1.0F, 10.0F, NAN};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, not_equal, lhs, rhs);

  xt::xarray<bool> expected = {true, false, true, true, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareNeS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> lhs = {min, min, min, 0, 0, 0, max, max, max};
  xt::xarray<int32_t> rhs = {min, 0, max, -1, 0, 1, min, 0, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, not_equal, lhs, rhs);

  xt::xarray<bool> expected = {false, true, true, true, false,
                               true,  true, true, false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareGeS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> lhs = {min, min, min, 0, 0, 0, max, max, max};
  xt::xarray<int32_t> rhs = {min, 0, max, -1, 0, 1, min, 0, max};

  auto ret = test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, greater_equal,
                                         lhs, rhs);

  xt::xarray<bool> expected = {true,  false, false, true, true,
                               false, true,  true,  true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareGtS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> lhs = {min, min, min, 0, 0, 0, max, max, max};
  xt::xarray<int32_t> rhs = {min, 0, max, -1, 0, 1, min, 0, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, greater, lhs, rhs);

  xt::xarray<bool> expected = {false, false, false, true, false,
                               false, true,  true,  false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareLeS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> lhs = {min, min, min, 0, 0, 0, max, max, max};
  xt::xarray<int32_t> rhs = {min, 0, max, -1, 0, 1, min, 0, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, less_equal, lhs, rhs);

  xt::xarray<bool> expected = {true, true,  true,  false, true,
                               true, false, false, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareLtS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> lhs = {min, min, min, 0, 0, 0, max, max, max};
  xt::xarray<int32_t> rhs = {min, 0, max, -1, 0, 1, min, 0, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, less, lhs, rhs);

  xt::xarray<bool> expected = {false, true,  true,  false, false,
                               true,  false, false, false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareEqU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> lhs = {0, 0, 0, 5, 5, 5, max, max, max};
  xt::xarray<uint32_t> rhs = {0, 1, max, 4, 5, 6, 0, 1, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, equal, lhs, rhs);

  xt::xarray<bool> expected = {true,  false, false, false, true,
                               false, false, false, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareNeU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> lhs = {0, 0, 0, 5, 5, 5, max, max, max};
  xt::xarray<uint32_t> rhs = {0, 1, max, 4, 5, 6, 0, 1, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, not_equal, lhs, rhs);

  xt::xarray<bool> expected = {false, true, true, true, false,
                               true,  true, true, false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareGeU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> lhs = {0, 0, 0, 5, 5, 5, max, max, max};
  xt::xarray<uint32_t> rhs = {0, 1, max, 4, 5, 6, 0, 1, max};

  auto ret = test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, greater_equal,
                                         lhs, rhs);

  xt::xarray<bool> expected = {true,  false, false, true, true,
                               false, true,  true,  true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareGtU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> lhs = {0, 0, 0, 5, 5, 5, max, max, max};
  xt::xarray<uint32_t> rhs = {0, 1, max, 4, 5, 6, 0, 1, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, greater, lhs, rhs);

  xt::xarray<bool> expected = {false, false, false, true, false,
                               false, true,  true,  false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareLeU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> lhs = {0, 0, 0, 5, 5, 5, max, max, max};
  xt::xarray<uint32_t> rhs = {0, 1, max, 4, 5, 6, 0, 1, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, less_equal, lhs, rhs);

  xt::xarray<bool> expected = {true, true,  true,  false, true,
                               true, false, false, true};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, CompareLtU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> lhs = {0, 0, 0, 5, 5, 5, max, max, max};
  xt::xarray<uint32_t> rhs = {0, 1, max, 4, 5, 6, 0, 1, max};

  auto ret =
      test::evalBinaryOp<uint8_t>(VIS_PUBLIC, VIS_PUBLIC, less, lhs, rhs);

  xt::xarray<bool> expected = {false, true,  true,  false, false,
                               true,  false, false, false};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, PowF32s) {
  xt::xarray<float> lhs = {0.0F, 4.0F, 2.0F, 2.0F, NAN, 6.0F, -2.0F, -2.0F};
  xt::xarray<float> rhs = {0.0F, 2.0F, -2.0F, 3.0F, 10.0F, NAN, 3.0F, 4.0F};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, power, lhs, rhs);

  xt::xarray<float> expected = {1.0F, 16.0F, 0.25F, 8.0F,
                                NAN,  NAN,   -8.0F, 16.0F};

  // FIXME: nan?
  // EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, PowNonIntegerF32s) {
  xt::xarray<float> lhs = {-2.0F, -0.6F, -0.6F, 0.0F};
  xt::xarray<float> rhs = {0.5F, 0.6F, -0.6F, -0.6F};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, power, lhs, rhs);

  xt::xarray<float> expected = {NAN, NAN, NAN, INFINITY};

  // FIXME: nan?
  // EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, MinF32s) {
  xt::xarray<float> lhs = {1.0F, 1.0F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {2.0F, -5.0F, 1.0F, 10.0F, NAN};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, min, lhs, rhs);

  xt::xarray<float> expected = {1.0F, -5.0F, 1.0F, NAN, NAN};

  // FIXME: nan?
  // EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, MinF64s) {
  xt::xarray<double> lhs = {1.0, 1.0, 2.25, NAN, 6.0};
  xt::xarray<double> rhs = {2.0, -5.0, 1.0, 10.0, NAN};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, min, lhs, rhs);

  xt::xarray<double> expected = {1.0, -5.0, 1.0, NAN, NAN};

  // FIXME: nan?
  // EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, MaxF32s) {
  xt::xarray<float> lhs = {1.0F, 1.0F, 2.25F, NAN, 6.0F};
  xt::xarray<float> rhs = {2.0F, -5.0F, 1.0F, 10.0F, NAN};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, max, lhs, rhs);

  xt::xarray<double> expected = {2.0F, 1.0F, 2.25F, NAN, NAN};

  // FIXME: nan?
  // EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, MaxF64s) {
  xt::xarray<double> lhs = {1.0, 1.0, 2.25, NAN, 6.0};
  xt::xarray<double> rhs = {2.0, -5.0, 1.0, 10.0, NAN};

  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, max, lhs, rhs);

  xt::xarray<double> expected = {2.0, 1.0, 2.25, NAN, NAN};

  // FIXME: nan?
  // EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, MaxS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> x = {min, min, min, -1, -1, 0, 0, 0, 1, 1, max, max, max};
  xt::xarray<int32_t> y = {min, max, 0, -10, 0, -1, 0, 1, 0, 10, 0, max, min};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, hal::max, x, y);

  xt::xarray<int32_t> expected = {min, max, 0,  -1,  0,   0,  0,
                                  1,   1,   10, max, max, max};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, MinS32s) {
  const int32_t min = std::numeric_limits<int32_t>::min();
  const int32_t max = std::numeric_limits<int32_t>::max();
  xt::xarray<int32_t> x = {min, min, min, -1, -1, 0, 0, 0, 1, 1, max, max, max};
  xt::xarray<int32_t> y = {min, max, 0, -10, 0, -1, 0, 1, 0, 10, 0, max, min};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, hal::min, x, y);

  xt::xarray<int32_t> expected = {min, min, min, -10, -1,  -1, 0,
                                  0,   0,   1,   0,   max, min};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, MaxU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> x = {0, 0, 1, 1, 1, max, max, max};
  xt::xarray<uint32_t> y = {0, 1, 0, 1, 10, 0, 234234, max};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, hal::max, x, y);

  xt::xarray<uint32_t> expected = {0, 1, 1, 1, 10, max, max, max};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, MinU32s) {
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  xt::xarray<uint32_t> x = {0, 0, 1, 1, 1, max, max, max};
  xt::xarray<uint32_t> y = {0, 1, 0, 1, 10, 0, 234234, max};

  auto ret =
      test::evalBinaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, hal::min, x, y);

  xt::xarray<uint32_t> expected = {0, 0, 0, 1, 1, 0, 234234, max};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, MaxTenF32s) {
  xt::xarray<float> x = {-0.0, 1.0, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0, 9.0};
  xt::xarray<float> y = {-0.0, -1.0, -2.0, 3.0, 4.0,
                         -5.0, -6.0, 7.0,  8.0, -9.0};
  auto ret = test::evalBinaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, hal::max, x, y);

  xt::xarray<float> expected = {-0.0, 1.0, 2.0, 3.0, 4.0,
                                5.0,  6.0, 7.0, 8.0, 9.0};

  EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, MinTenS32s) {
  xt::xarray<int32_t> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  xt::xarray<int32_t> y = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, hal::min, x, y);

  xt::xarray<int32_t> expected = {0, 1, 2, 3, 4, 4, 3, 2, 1, 0};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, MaxTenS32s) {
  xt::xarray<int32_t> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  xt::xarray<int32_t> y = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  auto ret =
      test::evalBinaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, hal::max, x, y);

  xt::xarray<int32_t> expected = {9, 8, 7, 6, 5, 5, 6, 7, 8, 9};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, NonNanClampF32) {
  xt::xarray<float> minimum = {1.0F, -6.5F, 1.0F, 2.25F, 0.0F};
  xt::xarray<float> argument = {2.0F, 10.0F, -5.0F, 1.0F, 10.0F};
  xt::xarray<float> maximum = {3.0F, 0.5F, 25.5F, 5.0F, 123.0};

  auto ret = test::evalTernaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, VIS_PUBLIC,
                                        clamp, minimum, argument, maximum);

  xt::xarray<float> expected = {2.0F, 0.5F, 1.0F, 2.25F, 10.0F};

  EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, ClampF32) {
  xt::xarray<float> minimum = {1.0F, -6.5F, 1.0F, 2.25F, NAN};
  xt::xarray<float> argument = {2.0F, 10.0F, -5.0F, 1.0F, 10.0F};
  xt::xarray<float> maximum = {3.0F, 0.5F, 25.5F, NAN, 123.0F};

  auto ret = test::evalTernaryOp<float>(VIS_PUBLIC, VIS_PUBLIC, VIS_PUBLIC,
                                        clamp, minimum, argument, maximum);

  xt::xarray<float> expected = {2.0F, 0.5F, 1.0F, NAN, NAN};

  // FIXME: nan?
  // EXPECT_TRUE(xt::allclose(expected, ret)) << ret << "\n" << expected;
}

TEST(ArrayElementwiseTest, ClampS32Vector) {
  xt::xarray<int32_t> min_vector = {1, -6, 1, 2, 0, -5};
  xt::xarray<int32_t> arg_vector = {2, 10, -5, 1, 4, 10};
  xt::xarray<int32_t> max_vector = {3, 0, 25, 5, 123, -1};

  auto ret =
      test::evalTernaryOp<int32_t>(VIS_PUBLIC, VIS_PUBLIC, VIS_PUBLIC, clamp,
                                   min_vector, arg_vector, max_vector);

  xt::xarray<int32_t> expected = {2, 0, 1, 2, 4, -1};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, ClampU32Vector) {
  xt::xarray<uint32_t> min_vector = {1, 2, 1, 2, 0, ~0U - 4};
  xt::xarray<uint32_t> arg_vector = {2, 10, 5, 1, 4, 10};
  xt::xarray<uint32_t> max_vector = {3, 5, 25, 5, 123, ~0U};

  auto ret =
      test::evalTernaryOp<uint32_t>(VIS_PUBLIC, VIS_PUBLIC, VIS_PUBLIC, clamp,
                                    min_vector, arg_vector, max_vector);

  xt::xarray<uint32_t> expected = {2, 5, 5, 2, 4, ~0U - 4};

  EXPECT_EQ(expected, ret);
}

TEST(ArrayElementwiseTest, ExpF32sVector) {
  // Just to help make sense of the scales here -- exp(89) saturates float32 and
  // exp(-10) is smaller than our error spec.
  xt::xarray<float> input_literal = {
      1.02,   -0.32,  0.85,   0.9,    1.23,   -0.91,  -0.49, 0.8,    -1.31,
      -1.44,  -0.13,  -1.31,  -0.79,  1.41,   1.21,   1.05,  -195.6, -194.5,
      -193.4, -192.3, -191.2, -190.1, -189.0, -187.9, -19.6, -18.5,  -17.4,
      -16.3,  -15.2,  -14.1,  -13.0,  -11.9,  -10.8,  -9.7,  -8.6,   -7.5,
      -6.4,   -5.3,   -4.2,   -3.1,   -2.0,   -0.9,   0.2,   1.3,    2.4,
      3.5,    4.6,    5.7,    6.8,    7.9,    9.0,    10.1,  11.2,   12.3,
      13.4,   14.5,   15.6,   16.7,   17.8,   18.9,   20.0,  21.1,   22.2,
      23.3,   24.4,   25.5,   26.6,   27.7,   28.8,   29.9,  31.0,   32.1,
      68.4,   69.5,   70.6,   71.7,   72.8,   73.9,   75.0,  76.1,   77.2,
      78.3,   79.4,   80.5,   81.6,   82.7,   83.8,   84.9,  85.2,   86.3,
      86.4,   86.5,   87.6,   87.7,   87.8,   87.9};

  auto ret = test::evalUnaryOp<float>(VIS_PUBLIC, exp, input_literal);

  std::vector<float> expected;
  for (size_t i = 0; i < input_literal.size(); i++) {
    expected.push_back(std::exp(input_literal[i]));
  }

  // FIXME: Improve exp accuracy
  // EXPECT_TRUE(xt::allclose(xt::adapt(expected), ret)) << ret << "\n"
  //                                                     << xt::adapt(expected);
}

TEST(ArrayElementwiseTest, LogF32sVector) {
  xt::xarray<float> input_literal = {
      -1.29,    -1.41,    -1.25,    -13.5,    -11.7,    -17.9,    -198,
      -167,     1.29,     1.41,     1.25,     13.5,     11.7,     17.9,
      198,      167,      1.27e+03, 1.33e+03, 1.74e+03, 1.6e+04,  1.84e+04,
      1.74e+04, 1.89e+05, 1.9e+05,  1.93e+06, 1.98e+06, 1.65e+06, 1.97e+07,
      1.66e+07, 1e+07,    1.98e+08, 1.96e+08, 1.64e+09, 1.58e+09, 1.64e+09,
      1.44e+10, 1.5e+10,  1.99e+10, 1.17e+11, 1.08e+11, 1.08e+12, 1.38e+12,
      1.4e+12,  1.03e+13, 1.6e+13,  1.99e+13, 1.26e+14, 1.51e+14, 1.33e+15,
      1.41e+15, 1.63e+15, 1.39e+16, 1.21e+16, 1.27e+16, 1.28e+17, 1.62e+17,
      2e+18,    1.96e+18, 1.81e+18, 1.99e+19, 1.86e+19, 1.61e+19, 1.71e+20,
      1.47e+20, 1.83e+21, 1.33e+21, 1.3e+21,  1.35e+22, 1.84e+22, 1.02e+22,
      1.81e+23, 1.02e+23, 1.89e+24, 1.49e+24, 1.08e+24, 1.95e+25, 1.1e+25,
      1.62e+25, 1.2e+26,  1.41e+26, 1.93e+27, 1.66e+27, 1.62e+27, 1.05e+28,
      1.5e+28,  1.79e+28, 1.36e+29, 1.95e+29, 1.5e+30,  1.81e+30, 1.34e+30,
      1.7e+31,  1.44e+31, 1.1e+31,  1.4e+32,  1.67e+32, 1.96e+33, 1.11e+33,
      1.19e+33, 1.61e+34, 1.05e+34, 1.88e+34, 1.67e+35, 1.7e+35};

  auto ret = test::evalUnaryOp<float>(VIS_PUBLIC, log, input_literal);

  std::vector<float> expected;
  for (size_t i = 0; i < input_literal.size(); i++) {
    expected.push_back(std::log(input_literal[i]));
  }

  // FIXME: Improve log accuracy
  //  EXPECT_TRUE(xt::allclose(xt::adapt(expected), ret)) << ret << "\n"
  //                                                      <<
  //                                                      xt::adapt(expected);
}

}  // namespace spu::kernel::hal
