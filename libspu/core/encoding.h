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

#pragma once

#include "libspu/core/array_ref.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type.h"

namespace spu {

// Encode an array to another array
//
// Given:
//   x: the source array
//   to_type: encoded storage type
//   fxp_bits: number of fractional bits for fixed point.
//
// Then:
//   let scale = 1 << fxp_bits
//   in
//   y = cast<to_type>(x * scale)   if type(x) is float
//    |= cast<to_type>(x)           if type(x) is integer
//   dtype = FXP if type(x) is float
//        |= INT if type(x) is integer
// NdArrayRef encodeToRing(const NdArrayRef& x, const Type& to_type,
//                        size_t fxp_bits, DataType* dtype = nullptr);

// Decode an array to another array
//
// Given:
//   x: the source array
//   to_type: the decoded storage type
//   fxp_bits: number of fractional bits for fixed point.
//   dtype: the source array data type.
//
// Then:
//   let scale = 1 << fxp_bits
//   in
//   y = cast<to_type>(x) / scale   if dtype is FXP
//    |= cast<to_type>(x)           if dtype is INT
// NdArrayRef decodeFromRing(const NdArrayRef& x, const Type& to_type,
//                          size_t fxp_bits, DataType dtype);

#define MAP_PTTYPE_TO_DTYPE(FN) \
  FN(PT_I8, DT_I8)              \
  FN(PT_U8, DT_U8)              \
  FN(PT_I16, DT_I16)            \
  FN(PT_U16, DT_U16)            \
  FN(PT_I32, DT_I32)            \
  FN(PT_U32, DT_U32)            \
  FN(PT_I64, DT_I64)            \
  FN(PT_U64, DT_U64)            \
  FN(PT_BOOL, DT_I1)            \
  FN(PT_F32, DT_F32)            \
  FN(PT_F64, DT_F64)

#define MAP_DTYPE_TO_PTTYPE(FN) \
  FN(DT_I8, PT_I8)              \
  FN(DT_U8, PT_U8)              \
  FN(DT_I16, PT_I16)            \
  FN(DT_U16, PT_U16)            \
  FN(DT_I32, PT_I32)            \
  FN(DT_U32, PT_U32)            \
  FN(DT_I64, PT_I64)            \
  FN(DT_U64, PT_U64)            \
  FN(DT_I1, PT_BOOL)            \
  FN(DT_F32, PT_F32)            \
  FN(DT_F64, PT_F64)

DataType getEncodeType(PtType pt_type);

PtType getDecodeType(DataType dtype);

// TODO: document me, verbosely
ArrayRef encodeToRing(const ArrayRef& src, FieldType field, size_t fxp_bits,
                      DataType* out_dtype = nullptr);
NdArrayRef encodeToRing(const NdArrayRef& src, FieldType field, size_t fxp_bits,
                        DataType* out_dtype = nullptr);

// TODO: document me, verbosely
ArrayRef decodeFromRing(const ArrayRef& src, DataType in_dtype, size_t fxp_bits,
                        PtType* out_pt_type = nullptr);
NdArrayRef decodeFromRing(const NdArrayRef& src, DataType in_dtype,
                          size_t fxp_bits, PtType* out_pt_type = nullptr);

}  // namespace spu
