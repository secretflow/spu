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

#include "libspu/core/encoding.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "libspu/core/type.h"
#include "libspu/core/xt_helper.h"

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
  EXPECT_EQ(getEncodeType(PT_I1), DT_I1);
  EXPECT_EQ(getEncodeType(PT_F32), DT_F32);
  EXPECT_EQ(getEncodeType(PT_F64), DT_F64);

  EXPECT_EQ(getDecodeType(DT_I8), PT_I8);
  EXPECT_EQ(getDecodeType(DT_U8), PT_U8);
  EXPECT_EQ(getDecodeType(DT_I16), PT_I16);
  EXPECT_EQ(getDecodeType(DT_U16), PT_U16);
  EXPECT_EQ(getDecodeType(DT_I32), PT_I32);
  EXPECT_EQ(getDecodeType(DT_U32), PT_U32);
  EXPECT_EQ(getDecodeType(DT_I64), PT_I64);
  EXPECT_EQ(getDecodeType(DT_U64), PT_U64);
  EXPECT_EQ(getDecodeType(DT_I1), PT_I1);
  EXPECT_EQ(getDecodeType(DT_F32), PT_F32);
  EXPECT_EQ(getDecodeType(DT_F64), PT_F64);
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

  // ring encoding
  {
    // GIVEN
    std::array<FloatT, 6> samples = {
        -std::numeric_limits<FloatT>::infinity(),
        std::numeric_limits<FloatT>::infinity(),
        -1.0,
        0.0,
        1.0,
        3.1415926,
    };

    NdArrayRef frm(makePtType(PtTypeToEnum<FloatT>::value), {samples.size()});
    std::copy(samples.begin(), samples.end(), &frm.at<FloatT>({0}));

    PtBufferView frm_pv(static_cast<const void*>(frm.data()),
                        PtTypeToEnum<FloatT>::value, frm.shape(),
                        frm.strides());

    DataType encoded_dtype_by_pv;
    auto encoded_by_pv =
        encodeToRing(frm_pv, kField, kFxpBits, &encoded_dtype_by_pv);

    if constexpr (std::is_same_v<FloatT, float>) {
      EXPECT_EQ(encoded_dtype_by_pv, DT_F32);
    } else {
      EXPECT_EQ(encoded_dtype_by_pv, DT_F64);
    }

    PtType out_pt_type_by_pv;
    NdArrayRef decoded_by_pv(makePtType(PtTypeToEnum<FloatT>::value),
                             {samples.size()});
    PtBufferView decoded_pv(static_cast<void*>(decoded_by_pv.data()),
                            PtTypeToEnum<FloatT>::value, decoded_by_pv.shape(),
                            decoded_by_pv.strides());
    decodeFromRing(encoded_by_pv, encoded_dtype_by_pv, kFxpBits, &decoded_pv,
                   &out_pt_type_by_pv);

    if constexpr (std::is_same_v<FloatT, float>) {
      EXPECT_EQ(out_pt_type_by_pv, PT_F32);
    } else {
      EXPECT_EQ(out_pt_type_by_pv, PT_F64);
    }
    auto* out_ptr_by_pv = &decoded_by_pv.at<FloatT>({0});
    const int64_t kReprBits = SizeOf(kField) * 8 - 2;
    const int64_t kScale = 1LL << kFxpBits;
    EXPECT_EQ(out_ptr_by_pv[0],
              -static_cast<FloatT>((1LL << kReprBits)) / kScale);
    EXPECT_EQ(out_ptr_by_pv[1],
              static_cast<FloatT>((1LL << kReprBits) - 1) / kScale);
    EXPECT_EQ(out_ptr_by_pv[2], -1.0);
    EXPECT_EQ(out_ptr_by_pv[3], 0.0);
    EXPECT_EQ(out_ptr_by_pv[4], 1.0);
    EXPECT_NEAR(out_ptr_by_pv[5], 3.1415926, 0.00001F);
  }
  // gfmp encoding
  {
    // GIVEN
    std::array<FloatT, 6> samples = {
        -std::numeric_limits<FloatT>::infinity(),
        std::numeric_limits<FloatT>::infinity(),
        -1.0,
        0.0,
        1.0,
        3.1415926,
    };

    NdArrayRef frm(makePtType(PtTypeToEnum<FloatT>::value), {samples.size()});
    std::copy(samples.begin(), samples.end(), &frm.at<FloatT>({0}));

    PtBufferView frm_pv(static_cast<const void*>(frm.data()),
                        PtTypeToEnum<FloatT>::value, frm.shape(),
                        frm.strides());

    DataType encoded_dtype_by_pv;
    auto encoded_by_pv =
        encodeToGfmp(frm_pv, kField, kFxpBits, &encoded_dtype_by_pv);

    if constexpr (std::is_same_v<FloatT, float>) {
      EXPECT_EQ(encoded_dtype_by_pv, DT_F32);
    } else {
      EXPECT_EQ(encoded_dtype_by_pv, DT_F64);
    }

    PtType out_pt_type_by_pv;
    NdArrayRef decoded_by_pv(makePtType(PtTypeToEnum<FloatT>::value),
                             {samples.size()});
    PtBufferView decoded_pv(static_cast<void*>(decoded_by_pv.data()),
                            PtTypeToEnum<FloatT>::value, decoded_by_pv.shape(),
                            decoded_by_pv.strides());
    decodeFromGfmp(encoded_by_pv, encoded_dtype_by_pv, kFxpBits, &decoded_pv,
                   &out_pt_type_by_pv);

    if constexpr (std::is_same_v<FloatT, float>) {
      EXPECT_EQ(out_pt_type_by_pv, PT_F32);
    } else {
      EXPECT_EQ(out_pt_type_by_pv, PT_F64);
    }
    auto* out_ptr_by_pv = &decoded_by_pv.at<FloatT>({0});
    const int64_t kReprBits = GetMersennePrimeExp(kField) - 1;
    const int64_t kScale = 1LL << kFxpBits;
    EXPECT_EQ(out_ptr_by_pv[0],
              -static_cast<FloatT>((1LL << kReprBits)) / kScale);
    EXPECT_EQ(out_ptr_by_pv[1],
              static_cast<FloatT>((1LL << kReprBits) - 1) / kScale);
    EXPECT_EQ(out_ptr_by_pv[2], -1.0);
    EXPECT_EQ(out_ptr_by_pv[3], 0.0);
    EXPECT_EQ(out_ptr_by_pv[4], 1.0);
    EXPECT_NEAR(out_ptr_by_pv[5], 3.1415926, 0.00001F);
  }
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

  // ring encoding
  {
    // GIVEN
    std::array<IntT, 6> samples = {
        std::numeric_limits<IntT>::min(),
        std::numeric_limits<IntT>::max(),
        static_cast<IntT>(-1),
        0,
        1,
    };

    NdArrayRef frm(makePtType(PtTypeToEnum<IntT>::value), {samples.size()});
    std::copy(samples.begin(), samples.end(), &frm.at<IntT>({0}));

    PtBufferView frm_pv(static_cast<const void*>(frm.data()),
                        PtTypeToEnum<IntT>::value, frm.shape(), frm.strides());

    DataType ring_encoded_dtype;
    auto ring_encoded_by_pv =
        encodeToRing(frm_pv, kField, kFxpBits, &ring_encoded_dtype);
    EXPECT_EQ(ring_encoded_dtype, getEncodeType(frm_pt_type));

    PtType out_pt_type;

    NdArrayRef decoded_by_pv(makePtType(PtTypeToEnum<IntT>::value),
                             {samples.size()});
    PtBufferView decoded_pv(static_cast<void*>(decoded_by_pv.data()),
                            PtTypeToEnum<IntT>::value, decoded_by_pv.shape(),
                            decoded_by_pv.strides());
    decodeFromRing(ring_encoded_by_pv, ring_encoded_dtype, kFxpBits,
                   &decoded_pv, &out_pt_type);
    EXPECT_EQ(out_pt_type, frm_pt_type);

    IntT* out_ptr_by_pv = &decoded_by_pv.at<IntT>({0});
    EXPECT_EQ(out_ptr_by_pv[0], samples[0]);
    EXPECT_EQ(out_ptr_by_pv[1], samples[1]);
    EXPECT_EQ(out_ptr_by_pv[2], static_cast<IntT>(-1));
    EXPECT_EQ(out_ptr_by_pv[3], 0);
    EXPECT_EQ(out_ptr_by_pv[4], 1);
  }
  // gfmp encoding
  {
    size_t mp_exp = GetMersennePrimeExp(kField);
    int128_t p = (static_cast<int128_t>(1) << mp_exp) - 1;
    int128_t max_positve = p >> 1;
    int128_t min_negetive = -max_positve;

    // clamp to the range of the Prime fie
    int128_t int_min = std::max(
        static_cast<int128_t>(std::numeric_limits<IntT>::min()), min_negetive);
    int128_t int_max = std::min(
        static_cast<int128_t>(std::numeric_limits<IntT>::max()), max_positve);

    // GIVEN
    std::array<IntT, 6> samples = {
        static_cast<IntT>(int_min),
        static_cast<IntT>(int_max),
        static_cast<IntT>(-1),
        0,
        1,
    };

    NdArrayRef frm(makePtType(PtTypeToEnum<IntT>::value), {samples.size()});
    std::copy(samples.begin(), samples.end(), &frm.at<IntT>({0}));

    PtBufferView frm_pv(static_cast<const void*>(frm.data()),
                        PtTypeToEnum<IntT>::value, frm.shape(), frm.strides());

    DataType gfmp_encoded_dtype;
    auto gfmp_encoded_by_pv =
        encodeToGfmp(frm_pv, kField, kFxpBits, &gfmp_encoded_dtype);

    EXPECT_EQ(gfmp_encoded_dtype, getEncodeType(frm_pt_type));

    PtType out_pt_type;

    NdArrayRef decoded_by_pv(makePtType(PtTypeToEnum<IntT>::value),
                             {samples.size()});
    PtBufferView decoded_pv(static_cast<void*>(decoded_by_pv.data()),
                            PtTypeToEnum<IntT>::value, decoded_by_pv.shape(),
                            decoded_by_pv.strides());
    decodeFromGfmp(gfmp_encoded_by_pv, gfmp_encoded_dtype, kFxpBits,
                   &decoded_pv, &out_pt_type);
    EXPECT_EQ(out_pt_type, frm_pt_type);

    IntT* out_ptr_by_pv = &decoded_by_pv.at<IntT>({0});

    EXPECT_EQ(out_ptr_by_pv[0], samples[0]);
    EXPECT_EQ(out_ptr_by_pv[1], samples[1]);
    EXPECT_EQ(out_ptr_by_pv[2], static_cast<IntT>(-1));
    EXPECT_EQ(out_ptr_by_pv[3], 0);
    EXPECT_EQ(out_ptr_by_pv[4], 1);
  }
}

}  // namespace spu
