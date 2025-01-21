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
// Datatype related
//////////////////////////////////////////////////////////////
#define CASE(DTYPE, _, __) \
  case DTYPE:              \
    return true;

bool isInteger(DataType dtype) {
  switch (dtype) {
    FOREACH_INT_DTYPES(CASE)
    default:
      return false;
  }
}

bool isFixedPoint(DataType dtype) {
  switch (dtype) {
    FOREACH_FXP_DTYPES(CASE)
    default:
      return false;
  }
}
#undef CASE

size_t getWidth(DataType dtype) {
#define CASE(DTYPE, _, LENGTH) \
  case DTYPE:                  \
    return LENGTH;

  switch (dtype) {
    FOREACH_DTYPES(CASE)
    default:
      SPU_THROW("invalid dtype {}", dtype);
  }

#undef CASE
}

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
#define CASE(DTYPE, SHORT_NAME, _) \
  case DTYPE:                      \
    os << #SHORT_NAME;             \
    break;

  switch (dtype) {
    FOREACH_DTYPES(CASE)
    default:
      // Unknown is a valid state.
      os << "*";
  }

#undef CASE
  return os;
}

//////////////////////////////////////////////////////////////
// Plaintext types.
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const PtType& pt_type) {
  os << PtType_Name(pt_type);
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

//////////////////////////////////////////////////////////////
// ProtocolKind utils
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, ProtocolKind protocol) {
  os << ProtocolKind_Name(protocol);
  return os;
}

//////////////////////////////////////////////////////////////
// Field GFP mappings, currently only support Mersenne primes
//////////////////////////////////////////////////////////////
size_t GetMersennePrimeExp(FieldType field) {
#define CASE(Name, ScalarT, MersennePrimeExp) \
  case FieldType::Name:                       \
    return MersennePrimeExp;                  \
    break;
  switch (field) {
    FIELD_TO_MERSENNE_PRIME_EXP_MAP(CASE)
    default:
      SPU_THROW("unknown supported field {}", field);
  }
#undef CASE
}

//////////////////////////////////////////////////////////////
// Field 2k types, TODO(jint) support Zq
//////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, FieldType field) {
  os << FieldType_Name(field);
  return os;
}

PtType GetStorageType(FieldType field) {
#define CASE(Name, StorageType) \
  case FieldType::Name:         \
    return StorageType;         \
    break;
  switch (field) {
    FIELD_TO_STORAGE_MAP(CASE)
    default:
      SPU_THROW("unknown storage type of {}", field);
  }
#undef CASE
}

FieldType PtTypeToField(PtType pt_type) {
#define CASE(FIELD_NAME, PT_NAME) \
  case PT_NAME:                   \
    return FieldType::FIELD_NAME;

  switch (pt_type) {
    FIELD_TO_STORAGE_MAP(CASE)
    default:
      SPU_THROW("can not convert pt_type={} to field", pt_type);
  }
#undef CASE
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
