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

// this file defines the basic types use in spu vm.

#pragma once

#include <complex>
#include <numeric>

#include "yacl/base/int128.h"

#include "libspu/core/half.h"
#include "libspu/core/prelude.h"

#include "libspu/spu.pb.h"

namespace spu {

// the rank definition, use for represent a unique party id.
using Rank = size_t;
constexpr Rank kInvalidRank = std::numeric_limits<Rank>::max();

//////////////////////////////////////////////////////////////
// Visibility related.
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const Visibility& vtype);

//////////////////////////////////////////////////////////////
// DataType related.
//////////////////////////////////////////////////////////////
// eq: hint: 数据类型绑定到SPU内建类型上。
/**
 * eq: 用于isInteger和getWidth。
 * FN是一个宏。SPU会在定义FN的宏内完成函数实现，然后解除对FN的宏定义。
*/
#define FOREACH_INT_DTYPES(FN) \
  FN(DT_I1, I1, 1)             \
  FN(DT_I8, I8, 8)             \
  FN(DT_U8, U8, 8)             \
  FN(DT_I16, I16, 16)          \
  FN(DT_U16, U16, 16)          \
  FN(DT_I32, I32, 32)          \
  FN(DT_U32, U32, 32)          \
  FN(DT_I64, I64, 64)          \
  FN(DT_U64, U64, 64)

/**
 * eq: 用于isFixedPoint和getWidth。
 * FN是一个宏。SPU会在定义FN的宏内完成函数实现，然后解除对FN的宏定义。
*/
#define FOREACH_FXP_DTYPES(FN) \
  FN(DT_F16, F16, 16)          \
  FN(DT_F32, F32, 32)          \
  FN(DT_F64, F64, 64)

#define FOREACH_DTYPES(FN) \
  FOREACH_INT_DTYPES(FN)   \
  FOREACH_FXP_DTYPES(FN)

bool isInteger(DataType dtype);
bool isFixedPoint(DataType dtype);
size_t getWidth(DataType dtype);

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

//////////////////////////////////////////////////////////////
// Plaintext c++ utilities
//////////////////////////////////////////////////////////////
// eq: hint: c++内建类型绑定到SPU内建类型上。
#define FOREACH_FLOAT_PT_TYPES(FN)  \
  FN(PT_F16, half_float::half, F16) \
  FN(PT_F32, float, F32)            \
  FN(PT_F64, double, F64)

#define FOREACH_INT_PT_TYPES(FN) \
  FN(PT_I8, int8_t, I8)          \
  FN(PT_U8, uint8_t, U8)         \
  FN(PT_I16, int16_t, I16)       \
  FN(PT_U16, uint16_t, U16)      \
  FN(PT_I32, int32_t, I32)       \
  FN(PT_U32, uint32_t, U32)      \
  FN(PT_I64, int64_t, I64)       \
  FN(PT_U64, uint64_t, U64)      \
  FN(PT_I128, int128_t, I128)    \
  FN(PT_U128, uint128_t, U128)   \
  FN(PT_I1, bool, I1)

#define FOREACH_COMPLEX_PT_TYPES(FN)     \
  FN(PT_CF32, std::complex<float>, CF32) \
  FN(PT_CF64, std::complex<double>, CF64)

#define FOREACH_PT_TYPES(FN) \
  FOREACH_INT_PT_TYPES(FN)   \
  FOREACH_FLOAT_PT_TYPES(FN)

// Helper macros to enumerate all py types.
// NOLINTNEXTLINE: Global internal used macro.

/**
 * eq: 分支语句，通过宏实现类型的编译期绑定。
 * 定义_kName，定义类型ScalarT，执行函数。
*/
#define __CASE_PT_TYPE(PT_TYPE, ...)             \
  case (PT_TYPE): {                              \
    using ScalarT = EnumToPtType<PT_TYPE>::type; \
    return __VA_ARGS__();                        \
  }

/**
 * eq: 针对浮点数明文类型的分发函数，根据pt_type类型调用不同的case分支。
 * 完成__CASE_PT_TYPE宏中的两种变量定义，最后执行函数。
 * 安全协议需要为DISPATCH_FLOAT_PT_TYPES提供lambda函数实现，以获得明文类型ScalarT支持。
*/
#define DISPATCH_FLOAT_PT_TYPES(PT_TYPE, ...)               \
  [&] {                                                     \
    switch (PT_TYPE) {                                      \
      __CASE_PT_TYPE(spu::PT_F16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F64, __VA_ARGS__)              \
      default:                                              \
        SPU_THROW("unimplemented for pt_type={}", PT_TYPE); \
    }                                                       \
  }()

/**
 * eq: 针对整数明文类型的分发函数，根据pt_type类型调用不同的case分支。
 * 完成__CASE_PT_TYPE宏中的两种变量定义，最后执行函数。
 * 安全协议需要为DISPATCH_UINT_PT_TYPES提供lambda函数实现，以获得明文类型ScalarT支持。
*/
#define DISPATCH_UINT_PT_TYPES(PT_TYPE, ...)                \
  [&] {                                                     \
    switch (PT_TYPE) {                                      \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U128, __VA_ARGS__)             \
      default:                                              \
        SPU_THROW("unimplemented for pt_type={}", PT_TYPE); \
    }                                                       \
  }()

/**
 * eq: 针对整数明文类型的分发函数，根据pt_type类型调用不同的case分支。
 * 完成__CASE_PT_TYPE宏中的两种变量定义，最后执行函数。
 * 安全协议需要为DISPATCH_INT_PT_TYPES提供lambda函数实现，以获得明文类型ScalarT支持。
 * 与DISPATCH_UINT_PT_TYPES的区别在于，多了PT_I1、PT_I8。
*/
#define DISPATCH_INT_PT_TYPES(PT_TYPE, ...)                 \
  [&] {                                                     \
    switch (PT_TYPE) {                                      \
      __CASE_PT_TYPE(spu::PT_I1, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_I8, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_I16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_I32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_I64, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)              \
      default:                                              \
        SPU_THROW("unimplemented for pt_type={}", PT_TYPE); \
    }                                                       \
  }()

#define DISPATCH_ALL_PT_TYPES(PT_TYPE, ...)                 \
  [&] {                                                     \
    switch (PT_TYPE) {                                      \
      __CASE_PT_TYPE(spu::PT_I1, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_I8, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_I16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_I32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_I64, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F64, __VA_ARGS__)              \
      default:                                              \
        SPU_THROW("unimplemented for pt_type={}", PT_TYPE); \
    }                                                       \
  }()

#define DISPATCH_ALL_NONE_BOOL_PT_TYPES(PT_TYPE, ...)       \
  [&] {                                                     \
    switch (PT_TYPE) {                                      \
      __CASE_PT_TYPE(spu::PT_I8, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)               \
      __CASE_PT_TYPE(spu::PT_I16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_I32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_I64, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F16, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F32, __VA_ARGS__)              \
      __CASE_PT_TYPE(spu::PT_F64, __VA_ARGS__)              \
      default:                                              \
        SPU_THROW("unimplemented for pt_type={}", PT_TYPE); \
    }                                                       \
  }()

std::ostream& operator<<(std::ostream& os, const PtType& pt_type);

size_t SizeOf(PtType ptt);

/**
 * eq: 结构体，用于实现物理类型（eg. uint64_t）到枚举（PtType）的映射。
 * 通过下文定义的宏实现模板特化。
*/
template <typename Type>
struct PtTypeToEnum {};

/**
 * eq: 结构体，用于实现枚举（PtType）到物理类型（eg. uint64_t）的映射。
 * 通过下文定义的宏实现模板特化。
*/
template <PtType Name>
struct EnumToPtType {};

/**
 * eq: 实现Name (Enum, PtType)和Type (real type, eg. uint64_t)的映射。
*/
#define CASE(Name, Type, _)               \
  template <>                             \
  struct PtTypeToEnum<Type> {             \
    static constexpr PtType value = Name; \
  };                                      \
                                          \
  template <>                             \
  struct EnumToPtType<Name> {             \
    typedef Type type;                    \
  };
FOREACH_PT_TYPES(CASE)
#undef CASE

//////////////////////////////////////////////////////////////
// ProtocolKind utils
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, ProtocolKind protocol);

/**
 * eq: 将FM32/64/128绑定到PT_U32/64/128上。
 * FN指代下文的DEF_TRAITS宏。
*/
//////////////////////////////////////////////////////////////
// Field 2k types, TODO(jint) support Zq
//////////////////////////////////////////////////////////////
#define FIELD_TO_STORAGE_MAP(FN) \
  FN(FM32, PT_U32)               \
  FN(FM64, PT_U64)               \
  FN(FM128, PT_U128)

template <FieldType ft>
struct Ring2kTrait {};

/**
 * eq: 定义结构体实现对Ring2kTrait的模板特化，用于组合。
 * 面向特定的域类型，提供scalar_t、kField、kStorageType三个成员。
 * Name指定域类型（FM32），StorageType指定存储类型（PT_U32）。
*/
#define DEF_TRAITS(Name, StorageType)                          \
  template <>                                                  \
  struct Ring2kTrait<FieldType::Name> {                        \
    using scalar_t = typename EnumToPtType<StorageType>::type; \
    constexpr static FieldType kField = FieldType::Name;       \
    constexpr static PtType kStorageType = StorageType;        \
  };
FIELD_TO_STORAGE_MAP(DEF_TRAITS)
#undef DEF_TRAITS

std::ostream& operator<<(std::ostream& os, FieldType field);

// 根据有限域类型，返回其存储类型的明文类型表示。
PtType GetStorageType(FieldType field);
FieldType PtTypeToField(PtType pt_type);
inline size_t SizeOf(FieldType field) { return SizeOf(GetStorageType(field)); }

// Helper macros to enumerate all fields
// NOLINTNEXTLINE: Global internal used macro.

/**
 * eq: case分支语句封装。
 * 做三件事：定义_kField（枚举类型）和_kName，
 * 定义ring2k_t类型便于后续调用，最后执行函数。
*/
#define __CASE_FIELD(FIELD, ...)                                      \
  case (FIELD): {                                                     \
    /* inject `_kField` & `_kName` for the continuation call */       \
    [[maybe_unused]] constexpr spu::FieldType _kField = FIELD;        \
    using ring2k_t [[maybe_unused]] = Ring2kTrait<_kField>::scalar_t; \
    return __VA_ARGS__();                                             \
  }

/**
 * eq: 分发函数，根据field类型调用不同的case分支。主要用于算术分享计算。
 * 完成__CASE_FIELD宏中的两种变量定义、一种类型定义（ring2k_t，绑定到物理类型），并执行函数。
 * 安全协议需要为DISPATCH_ALL_FIELDS提供lambda函数实现，以获得ring2k_t。
*/
#define DISPATCH_ALL_FIELDS(FIELD, ...)                 \
  [&] {                                                 \
    switch (FIELD) {                                    \
      __CASE_FIELD(spu::FieldType::FM32, __VA_ARGS__)   \
      __CASE_FIELD(spu::FieldType::FM64, __VA_ARGS__)   \
      __CASE_FIELD(spu::FieldType::FM128, __VA_ARGS__)  \
      default:                                          \
        SPU_THROW("unimplemented for field={}", FIELD); \
    }                                                   \
  }()

//////////////////////////////////////////////////////////////
// Value range information, should it be here, at top level(jint)?
//////////////////////////////////////////////////////////////
enum class SignType {
  Unknown,
  Positive,
  Negative,
};
std::ostream& operator<<(std::ostream& os, const SignType& sign);

}  // namespace spu
