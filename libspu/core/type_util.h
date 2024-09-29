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
// Plaintext c++ utilities
//////////////////////////////////////////////////////////////
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
  FN(PT_I1, bool, I1)

#define FOREACH_COMPLEX_PT_TYPES(FN)     \
  FN(PT_CF32, std::complex<float>, CF32) \
  FN(PT_CF64, std::complex<double>, CF64)

#define FOREACH_PT_TYPES(FN) \
  FOREACH_INT_PT_TYPES(FN)   \
  FOREACH_FLOAT_PT_TYPES(FN)

// Helper macros to enumerate all py types.
// NOLINTNEXTLINE: Global internal used macro.
#define __CASE_PT_TYPE(PT_TYPE, ...)             \
  case (PT_TYPE): {                              \
    using ScalarT = EnumToPtType<PT_TYPE>::type; \
    return __VA_ARGS__();                        \
  }

#define DISPATCH_FLOAT_PT_TYPES(PT_TYPE, ...)                 \
  [&] {                                                       \
    switch (PT_TYPE) {                                        \
      __CASE_PT_TYPE(spu::PT_F16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F64, __VA_ARGS__)                \
      default:                                                \
        SPU_THROW("not implemented for pt_type={}", PT_TYPE); \
    }                                                         \
  }()

#define DISPATCH_UINT_PT_TYPES(PT_TYPE, ...)                  \
  [&] {                                                       \
    switch (PT_TYPE) {                                        \
      __CASE_PT_TYPE(spu::PT_I1, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)                \
      default:                                                \
        SPU_THROW("not implemented for pt_type={}", PT_TYPE); \
    }                                                         \
  }()

#define DISPATCH_SINT_PT_TYPES(PT_TYPE, ...)                  \
  [&] {                                                       \
    switch (PT_TYPE) {                                        \
      __CASE_PT_TYPE(spu::PT_I1, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_I8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_I16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I128, __VA_ARGS__)               \
      default:                                                \
        SPU_THROW("not implemented for pt_type={}", PT_TYPE); \
    }                                                         \
  }()

#define DISPATCH_INT_PT_TYPES(PT_TYPE, ...)                   \
  [&] {                                                       \
    switch (PT_TYPE) {                                        \
      __CASE_PT_TYPE(spu::PT_I1, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_I8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_I16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I128, __VA_ARGS__)               \
      default:                                                \
        SPU_THROW("not implemented for pt_type={}", PT_TYPE); \
    }                                                         \
  }()

#define DISPATCH_ALL_PT_TYPES(PT_TYPE, ...)                   \
  [&] {                                                       \
    switch (PT_TYPE) {                                        \
      __CASE_PT_TYPE(spu::PT_I1, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_I8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_I16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I128, __VA_ARGS__)               \
      default:                                                \
        SPU_THROW("not implemented for pt_type={}", PT_TYPE); \
    }                                                         \
  }()

#define DISPATCH_ALL_NONE_BOOL_PT_TYPES(PT_TYPE, ...)         \
  [&] {                                                       \
    switch (PT_TYPE) {                                        \
      __CASE_PT_TYPE(spu::PT_I8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_U8, __VA_ARGS__)                 \
      __CASE_PT_TYPE(spu::PT_I16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_I64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_U64, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F16, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F32, __VA_ARGS__)                \
      __CASE_PT_TYPE(spu::PT_F64, __VA_ARGS__)                \
      default:                                                \
        SPU_THROW("not implemented for pt_type={}", PT_TYPE); \
    }                                                         \
  }()

// numpy type naming:
// https://numpy.org/doc/stable/reference/arrays.scalars.html#sized-aliases
#define FOR_PY_FORMATS(FN) \
  FN("int8", PT_I8)        \
  FN("int16", PT_I16)      \
  FN("int32", PT_I32)      \
  FN("int64", PT_I64)      \
  FN("uint8", PT_U8)       \
  FN("uint16", PT_U16)     \
  FN("uint32", PT_U32)     \
  FN("uint64", PT_U64)     \
  FN("float16", PT_F16)    \
  FN("float32", PT_F32)    \
  FN("float64", PT_F64)    \
  FN("bool", PT_I1)        \
  FN("complex64", PT_CF32) \
  FN("complex128", PT_CF64)

spu::PtType PyFormatToPtType(const std::string& format);
std::string PtTypeToPyFormat(PtType pt_type);

std::ostream& operator<<(std::ostream& os, const PtType& pt_type);
std::ostream& operator<<(std::ostream& os, const SemanticType& pt_type);
std::ostream& operator<<(std::ostream& os, const StorageType& pt_type);

size_t SizeOf(PtType ptt);
size_t SizeOf(StorageType rst);
size_t SizeOf(SemanticType rse);

bool isUnsigned(SemanticType type);
SemanticType promoteToNextSignedType(SemanticType type);

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

// Ring Storage dispatch
#define FOREACH_RING_STORAGE_TYPES(FN) \
  FN(ST_8, uint8_t)                    \
  FN(ST_16, uint16_t)                  \
  FN(ST_32, uint32_t)                  \
  FN(ST_64, uint64_t)                  \
  FN(ST_128, uint128_t)

template <StorageType Name>
struct EnumToStorage {};

#define CASE(Name, Type)       \
  template <>                  \
  struct EnumToStorage<Name> { \
    typedef Type type;         \
  };
FOREACH_RING_STORAGE_TYPES(CASE)
#undef CASE

#define __CASE_ST_TYPE(ST_TYPE, ...)              \
  case (ST_TYPE): {                               \
    using ScalarT = EnumToStorage<ST_TYPE>::type; \
    return __VA_ARGS__();                         \
  }

#define DISPATCH_ALL_STORAGE_TYPES(RS_TYPE, ...)                      \
  [&] {                                                               \
    switch (RS_TYPE) {                                                \
      __CASE_ST_TYPE(spu::ST_8, __VA_ARGS__)                          \
      __CASE_ST_TYPE(spu::ST_16, __VA_ARGS__)                         \
      __CASE_ST_TYPE(spu::ST_32, __VA_ARGS__)                         \
      __CASE_ST_TYPE(spu::ST_64, __VA_ARGS__)                         \
      __CASE_ST_TYPE(spu::ST_128, __VA_ARGS__)                        \
      default:                                                        \
        SPU_THROW("unimplemented for ring storage type={}", RS_TYPE); \
    }                                                                 \
  }()

//////////////////////////////////////////////////////////////
// ProtocolKind utils
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, ProtocolKind protocol);

StorageType GetStorageType(size_t nbits);
SemanticType GetEncodedType(PtType type, size_t ring_width = 64);
SemanticType GetPlainTextSemanticType(PtType type);
SemanticType GetSemanticType(int64_t field);

// Get the minimum storage type for a ring2^k value given k = field.
inline size_t SizeOf(size_t field) { return SizeOf(GetStorageType(field)); }

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
