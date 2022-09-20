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

#include <numeric>

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "yasl/base/exception.h"
#include "yasl/base/int128.h"

#include "spu/spu.pb.h"

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

#define FOREACH_FXP_DTYPES(FN) FN(DT_FXP, FXP, 64)

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
#define FOREACH_FLOAT_PT_TYPES(FN) \
  FN(PT_F32, float, F32)           \
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
  FN(PT_BOOL, bool, I1)

#define FOREACH_PT_TYPES(FN) \
  FOREACH_INT_PT_TYPES(FN)   \
  FOREACH_FLOAT_PT_TYPES(FN)

// Helper macros to enumerate all py types.
#define __CASE_PT_TYPE(PT_TYPE, NAME, ...)                     \
  case (PT_TYPE): {                                            \
    [[maybe_unused]] constexpr std::string_view _kName = NAME; \
    using ScalarT = EnumToPtType<PT_TYPE>::type;               \
    return __VA_ARGS__();                                      \
  }

#define DISPATCH_FLOAT_PT_TYPES(PT_TYPE, NAME, ...)                      \
  [&] {                                                                  \
    switch (PT_TYPE) {                                                   \
      __CASE_PT_TYPE(spu::PT_F32, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_F64, NAME, __VA_ARGS__)                     \
      default:                                                           \
        YASL_THROW("{} not implemented for pt_type={}", #NAME, PT_TYPE); \
    }                                                                    \
  }()

#define DISPATCH_UINT_PT_TYPES(PT_TYPE, NAME, ...)                       \
  [&] {                                                                  \
    switch (PT_TYPE) {                                                   \
      __CASE_PT_TYPE(spu::PT_U8, NAME, __VA_ARGS__)                      \
      __CASE_PT_TYPE(spu::PT_U16, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U32, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U64, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U128, NAME, __VA_ARGS__)                    \
      default:                                                           \
        YASL_THROW("{} not implemented for pt_type={}", #NAME, PT_TYPE); \
    }                                                                    \
  }()

#define DISPATCH_INT_PT_TYPES(PT_TYPE, NAME, ...)                        \
  [&] {                                                                  \
    switch (PT_TYPE) {                                                   \
      __CASE_PT_TYPE(spu::PT_BOOL, NAME, __VA_ARGS__)                    \
      __CASE_PT_TYPE(spu::PT_I8, NAME, __VA_ARGS__)                      \
      __CASE_PT_TYPE(spu::PT_U8, NAME, __VA_ARGS__)                      \
      __CASE_PT_TYPE(spu::PT_I16, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U16, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_I32, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U32, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_I64, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U64, NAME, __VA_ARGS__)                     \
      default:                                                           \
        YASL_THROW("{} not implemented for pt_type={}", #NAME, PT_TYPE); \
    }                                                                    \
  }()

#define DISPATCH_ALL_PT_TYPES(PT_TYPE, NAME, ...)                        \
  [&] {                                                                  \
    switch (PT_TYPE) {                                                   \
      __CASE_PT_TYPE(spu::PT_BOOL, NAME, __VA_ARGS__)                    \
      __CASE_PT_TYPE(spu::PT_I8, NAME, __VA_ARGS__)                      \
      __CASE_PT_TYPE(spu::PT_U8, NAME, __VA_ARGS__)                      \
      __CASE_PT_TYPE(spu::PT_I16, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U16, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_I32, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U32, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_I64, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_U64, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_F32, NAME, __VA_ARGS__)                     \
      __CASE_PT_TYPE(spu::PT_F64, NAME, __VA_ARGS__)                     \
      default:                                                           \
        YASL_THROW("{} not implemented for pt_type={}", #NAME, PT_TYPE); \
    }                                                                    \
  }()

std::ostream& operator<<(std::ostream& os, const PtType& pt_type);

size_t SizeOf(PtType ptt);

template <typename Type>
struct PtTypeToEnum {};

template <PtType Name>
struct EnumToPtType {};

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

//////////////////////////////////////////////////////////////
// Field 2k types, TODO(jint) support Zq
//////////////////////////////////////////////////////////////
#define FIELD_TO_STORAGE_MAP(FN) \
  FN(FM32, PT_U32)               \
  FN(FM64, PT_U64)               \
  FN(FM128, PT_U128)

template <FieldType ft>
struct Ring2kTrait {};

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

PtType GetStorageType(FieldType field);
FieldType PtTypeToField(PtType pt_type);
inline size_t SizeOf(FieldType field) { return SizeOf(GetStorageType(field)); }

// Helper macros to enumerate all fields
#define __CASE_FIELD(FIELD, NAME, ...)                                \
  case (FIELD): {                                                     \
    /* inject `_kField` & `_kName` for the continuation call */       \
    [[maybe_unused]] constexpr spu::FieldType _kField = FIELD;        \
    [[maybe_unused]] constexpr std::string_view _kName = NAME;        \
    using ring2k_t [[maybe_unused]] = Ring2kTrait<_kField>::scalar_t; \
    return __VA_ARGS__();                                             \
  }

#define DISPATCH_ALL_FIELDS(FIELD, NAME, ...)                        \
  [&] {                                                              \
    switch (FIELD) {                                                 \
      __CASE_FIELD(spu::FieldType::FM32, NAME, __VA_ARGS__)          \
      __CASE_FIELD(spu::FieldType::FM64, NAME, __VA_ARGS__)          \
      __CASE_FIELD(spu::FieldType::FM128, NAME, __VA_ARGS__)         \
      default:                                                       \
        YASL_THROW("{} not implemented for field={}", #NAME, FIELD); \
    }                                                                \
  }()

// Return the default number of fractional bits.
size_t getDefaultFxpBits(const RuntimeConfig& config);

}  // namespace spu
