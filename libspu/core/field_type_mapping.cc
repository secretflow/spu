// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/core/field_type_mapping.h"

// The complexity expression library
namespace spu {

FieldType getFieldTypeFromDataType(DataType dtype) {
  switch (dtype) {
    case DT_I1:
      [[fallthrough]];
    case DT_I8:
      [[fallthrough]];
    case DT_U8:
      [[fallthrough]];
    case DT_I16:
      [[fallthrough]];
    case DT_U16:
      [[fallthrough]];
    case DT_I32:
      [[fallthrough]];
    case DT_U32:
      return FM32;
    case DT_I64:
      [[fallthrough]];
    case DT_U64:
      return FM64;
    case DT_F16:
      return FM32;
    case DT_F32:
      return FM64;
    case DT_F64:
      return FM128;
    default:
      SPU_THROW("Should not reach, {}", DataType_Name(dtype));
  }
}

FieldType getFieldFromPlainTextType(PtType pt_type) {
  switch (pt_type) {
    case PT_BOOL:
      [[fallthrough]];
    case PT_I8:
      [[fallthrough]];
    case PT_U8:
      [[fallthrough]];
    case PT_I16:
      [[fallthrough]];
    case PT_U16:
      [[fallthrough]];
    case PT_I32:
      [[fallthrough]];
    case PT_U32:
      return FM32;
    case PT_I64:
      [[fallthrough]];
    case PT_U64:
      return FM64;
    case PT_F16:
      return FM32;
    case PT_F32:
      return FM64;
    case PT_F64:
      return FM128;
    default:
      SPU_THROW("Should not reach, {}", PtType_Name(pt_type));
  }
}

DataType getIntegerTypeFromFieldType(FieldType field) {
  switch (field) {
    case FM32:
      return DT_I32;
    default:
      return DT_I64;
  }
}

DataType getUIntegerTypeFromFieldType(FieldType field) {
  switch (field) {
    case FM32:
      return DT_U32;
    default:
      return DT_U64;
  }
}

DataType getFxpFromFieldType(FieldType field) {
  switch (field) {
    case FM32:
      return DT_F16;
    case FM64:
      return DT_F32;
    default:
      return DT_F64;
  }
}

}  // namespace spu
