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

#include "libspu/core/type_util.h"

namespace spu {

//////////////////////////////////////////////////////////////
// Visibility related
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const Visibility& vtype) {
  switch (vtype) {
    case VIS_PUBLIC:
      os << "P";
      break;
    case VIS_SECRET:
      os << "S";
      break;
    case VIS_PRIVATE:
      os << "V";
      break;
    default:
      os << "Invalid";
  }
  return os;
}

//////////////////////////////////////////////////////////////
// Plaintext types.
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const PtType& pt_type) {
  os << PtType_Name(pt_type);
  return os;
}

std::ostream& operator<<(std::ostream& os, const StorageType& val) {
  os << StorageType_Name(val);
  return os;
}

std::ostream& operator<<(std::ostream& os, const SemanticType& val) {
  os << SemanticType_Name(val);
  return os;
}

size_t SizeOf(PtType ptt) {
#define CASE(Name, Type, _) \
  case (Name):              \
    return sizeof(Type);
  switch (ptt) {
    case PT_INVALID:
      return 0;
      FOREACH_PT_TYPES(CASE);
      FOREACH_COMPLEX_PT_TYPES(CASE);
    default:
      SPU_THROW("unknown size of {}", ptt);
  }
#undef CASE
}

size_t SizeOf(StorageType rst) {
  switch (rst) {
    case ST_8:
      return 1;
    case ST_16:
      return 2;
    case ST_32:
      return 4;
    case ST_64:
      return 8;
    case ST_128:
      return 16;
    default:
      SPU_THROW("unknown sizeof {}", rst);
  }
}

size_t SizeOf(SemanticType rse) {
  switch (rse) {
    case SE_1:
    case SE_I8:
    case SE_U8:
      return 1;
    case SE_I16:
    case SE_U16:
      return 2;
    case SE_I32:
    case SE_U32:
      return 4;
    case SE_I64:
    case SE_U64:
      return 8;
    case SE_I128:
      return 16;
    default:
      SPU_THROW("unknown sizeof {}", rse);
  }
}

bool isUnsigned(SemanticType type) {
  switch (type) {
    case SE_1:
      return false;
    case SE_I8:
    case SE_I16:
    case SE_I32:
    case SE_I64:
    case SE_I128:
      return false;
    default:
      return true;
  }
}

SemanticType promoteToNextSignedType(SemanticType type) {
  switch (type) {
    case SE_1:
      return SE_1;
    case SE_I8:
    case SE_U8:
      return SE_I16;
    case SE_I16:
    case SE_U16:
      return SE_I32;
    case SE_I32:
    case SE_U32:
      return SE_I64;
    case SE_I64:
    case SE_U64:
      return SE_I128;
    case SE_I128:
      return SE_I128;
    default:
      SPU_THROW("Cannot find next signed type for {}", type);
  }
}

// https://docs.python.org/3/library/struct.html#format-characters
// https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
// Note: python and numpy has different type string, here pybind11 uses numpy's
// definition
spu::PtType PyFormatToPtType(const std::string& format) {
#define CASE(FORMAT, PT_TYPE) \
  if (format == (FORMAT)) return PT_TYPE;

  if (false) {  // NOLINT: macro trick
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  SPU_THROW("unknown py format={}", format);
}

std::string PtTypeToPyFormat(spu::PtType pt_type) {
#define CASE(FORMAT, PT_TYPE) \
  if (pt_type == (PT_TYPE)) return FORMAT;

  if (false) {  // NOLINT: macro trick
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  SPU_THROW("unknown pt_type={}", pt_type);
}

//////////////////////////////////////////////////////////////
// ProtocolKind utils
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, ProtocolKind protocol) {
  os << ProtocolKind_Name(protocol);
  return os;
}

// Get the minimum storage type nbits.
StorageType GetStorageType(size_t nbits) {
  SPU_ENFORCE(nbits <= 128, "unsupported storage type for {} bits", nbits);
  if (nbits > 64) return ST_128;
  if (nbits > 32) return ST_64;
  if (nbits > 16) return ST_32;
  if (nbits > 8) return ST_16;
  return ST_8;
}

SemanticType GetSemanticType(int64_t field) {
  switch (field) {
    case 8:
      return SE_I8;
    case 16:
      return SE_I16;
    case 32:
      return SE_I32;
    case 64:
      return SE_I64;
    case 128:
      return SE_I128;
  }
  SPU_THROW("unsupported field={}", field);
}

SemanticType GetEncodedType(PtType type, size_t ring_width) {
  switch (type) {
    case PT_I1:
      return SE_1;
    case PT_I8:
      return SE_I8;
    case PT_U8:
      return SE_U8;
    case PT_I16:
      return SE_I16;
    case PT_U16:
      return SE_U16;
    case PT_I32:
      return SE_I32;
    case PT_U32:
      return SE_U32;
    case PT_U64:
      return SE_U64;
    case PT_I64:
    case PT_F16:
    case PT_F32:
    case PT_F64:
      return ring_width == 64 ? SE_I64 : SE_I128;
    case PT_I128:
      return SE_I128;
    default:
      return SE_INVALID;
  }
}

SemanticType GetPlainTextSemanticType(PtType type) {
  switch (type) {
    case PT_I1:
      return SE_1;
    case PT_I8:
      return SE_I8;
    case PT_U8:
      return SE_U8;
    case PT_I16:
      return SE_I16;
    case PT_U16:
      return SE_U16;
    case PT_I32:
      return SE_I32;
    case PT_U32:
      return SE_U32;
    case PT_U64:
      return SE_U64;
    case PT_I64:
      return SE_I64;
    case PT_F16:
      return SE_I16;
    case PT_F32:
      return SE_I32;
    case PT_F64:
      return SE_I64;
    case PT_I128:
      return SE_I128;
    default:
      return SE_INVALID;
  }
}

//////////////////////////////////////////////////////////////
// SignType related
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const SignType& sign) {
  switch (sign) {
    case SignType::Positive:
      os << "Positive";
      break;
    case SignType::Negative:
      os << "Negative";
      break;
    case SignType::Unknown:
      os << "Unknown";
      break;
    default:
      os << "Invalid";
  }
  return os;
}

}  // namespace spu
