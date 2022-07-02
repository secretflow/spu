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

#include "spu/core/encoding.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "spu/core/type.h"
#include "spu/core/xt_helper.h"

namespace spu {

TEST(EncodingTypeTest, EncodeDecodeMap) {
  EXPECT_EQ(getEncodeType(PT_I8), DT_I8);
  EXPECT_EQ(getEncodeType(PT_U8), DT_U8);
  EXPECT_EQ(getEncodeType(PT_I16), DT_I16);
  EXPECT_EQ(getEncodeType(PT_U16), DT_U16);
  EXPECT_EQ(getEncodeType(PT_I32), DT_I32);
  EXPECT_EQ(getEncodeType(PT_U32), DT_U32);
  EXPECT_EQ(getEncodeType(PT_I64), DT_I64);
  EXPECT_EQ(getEncodeType(PT_U64), DT_U64);
  EXPECT_EQ(getEncodeType(PT_BOOL), DT_I1);
  EXPECT_EQ(getEncodeType(PT_F32), DT_FXP);
  EXPECT_EQ(getEncodeType(PT_F64), DT_FXP);

  EXPECT_EQ(getDecodeType(DT_I8), PT_I8);
  EXPECT_EQ(getDecodeType(DT_U8), PT_U8);
  EXPECT_EQ(getDecodeType(DT_I16), PT_I16);
  EXPECT_EQ(getDecodeType(DT_U16), PT_U16);
  EXPECT_EQ(getDecodeType(DT_I32), PT_I32);
  EXPECT_EQ(getDecodeType(DT_U32), PT_U32);
  EXPECT_EQ(getDecodeType(DT_I64), PT_I64);
  EXPECT_EQ(getDecodeType(DT_U64), PT_U64);
  EXPECT_EQ(getDecodeType(DT_I1), PT_BOOL);
  EXPECT_EQ(getDecodeType(DT_FXP), PT_F32);
}

using Field64 = std::integral_constant<FieldType, FM64>;
using Field128 = std::integral_constant<FieldType, FM128>;

using IntTypes = ::testing::Types<
    // <PtType, Field>
    std::tuple<bool, Field64>,       //
    std::tuple<int8_t, Field64>,     //
    std::tuple<uint8_t, Field64>,    //
    std::tuple<int16_t, Field64>,    //
    std::tuple<uint16_t, Field64>,   //
    std::tuple<int32_t, Field64>,    //
    std::tuple<uint32_t, Field64>,   //
    std::tuple<int64_t, Field64>,    //
    std::tuple<uint64_t, Field64>,   //
    std::tuple<bool, Field128>,      //
    std::tuple<int8_t, Field128>,    //
    std::tuple<uint8_t, Field128>,   //
    std::tuple<int16_t, Field128>,   //
    std::tuple<uint16_t, Field128>,  //
    std::tuple<int32_t, Field128>,   //
    std::tuple<uint32_t, Field128>,  //
    std::tuple<int64_t, Field128>,   //
    std::tuple<uint64_t, Field128>   //
    >;

using FloatTypes = ::testing::Types<
    // <PtType, Field>
    std::tuple<float, Field64>,  //
    std::tuple<double, Field64>  //
    // std::tuple<float, Field128>,  // FIXME: infinite test failed.
    // std::tuple<double, Field128>  // FIXME: infinite test failed.
    >;

template <typename S>
class FloatEncodingTest : public ::testing::Test {};
TYPED_TEST_SUITE(FloatEncodingTest, FloatTypes);

TYPED_TEST(FloatEncodingTest, Works) {
  using FloatT = typename std::tuple_element<0, TypeParam>::type;
  using FieldT = typename std::tuple_element<1, TypeParam>::type;
  constexpr FieldType kField = FieldT();
  constexpr size_t kFxpBits = 18;

  // GIVEN
  std::array<FloatT, 6> samples = {
      -std::numeric_limits<FloatT>::infinity(),
      std::numeric_limits<FloatT>::infinity(),
      -1.0,
      0.0,
      1.0,
      3.1415926,
  };

  ArrayRef frm(makePtType(PtTypeToEnum<FloatT>::value), samples.size());
  std::copy(samples.begin(), samples.end(), &frm.at<FloatT>(0));

  // std::cout << frm.at<FloatT>(0) << std::endl;

  DataType encoded_dtype;
  auto encoded = encodeToRing(frm, kField, kFxpBits, &encoded_dtype);
  EXPECT_EQ(encoded_dtype, DT_FXP);

  PtType out_pt_type;
  auto decoded = decodeFromRing(encoded, encoded_dtype, kFxpBits, &out_pt_type);
  EXPECT_EQ(out_pt_type, PT_F32);

  float* out_ptr = &decoded.at<float>(0);
  const int64_t kReprBits = SizeOf(kField) * 8 - 2;
  const int64_t kScale = 1LL << kFxpBits;
  EXPECT_EQ(out_ptr[0], -static_cast<float>((1LL << kReprBits)) / kScale);
  EXPECT_EQ(out_ptr[1], static_cast<float>((1LL << kReprBits) - 1) / kScale);
  EXPECT_EQ(out_ptr[2], -1.0);
  EXPECT_EQ(out_ptr[3], 0.0);
  EXPECT_EQ(out_ptr[4], 1.0);
  EXPECT_NEAR(out_ptr[5], 3.1415926, 0.00001F);
}

template <typename S>
class IntEncodingTest : public ::testing::Test {};
TYPED_TEST_SUITE(IntEncodingTest, IntTypes);

TYPED_TEST(IntEncodingTest, Works) {
  using IntT = typename std::tuple_element<0, TypeParam>::type;
  using FieldT = typename std::tuple_element<1, TypeParam>::type;
  constexpr PtType frm_pt_type = PtTypeToEnum<IntT>::value;
  constexpr FieldType kField = FieldT();
  constexpr size_t kFxpBits = 18;

  // GIVEN
  std::array<IntT, 6> samples = {
      std::numeric_limits<IntT>::min(),
      std::numeric_limits<IntT>::max(),
      static_cast<IntT>(-1),
      0,
      1,
  };

  ArrayRef frm(makePtType(PtTypeToEnum<IntT>::value), samples.size());
  std::copy(samples.begin(), samples.end(), &frm.at<IntT>(0));

  // std::cout << frm.at<IntT>(0) << std::endl;

  DataType encoded_dtype;
  auto encoded = encodeToRing(frm, kField, kFxpBits, &encoded_dtype);
  EXPECT_EQ(encoded_dtype, getEncodeType(frm_pt_type));

  PtType out_pt_type;
  auto decoded = decodeFromRing(encoded, encoded_dtype, kFxpBits, &out_pt_type);
  EXPECT_EQ(out_pt_type, frm_pt_type);

  IntT* out_ptr = &decoded.at<IntT>(0);
  EXPECT_EQ(out_ptr[0], samples[0]);
  EXPECT_EQ(out_ptr[1], samples[1]);
  EXPECT_EQ(out_ptr[2], static_cast<IntT>(-1));
  EXPECT_EQ(out_ptr[3], 0);
  EXPECT_EQ(out_ptr[4], 1);
}

}  // namespace spu
