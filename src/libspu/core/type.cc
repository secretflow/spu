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

#include "libspu/core/type.h"

#include <mutex>

#include "magic_enum.hpp"

namespace spu {

Type::Type()
    : model_(std::make_unique<VoidTy>()), cached_model_size_(model_->size()) {}

Type::Type(std::unique_ptr<TypeObject> model)
    : model_(std::move(model)), cached_model_size_(model_->size()) {}

Type& Type::operator=(const Type& other) {
  model_ = other.model_->clone();
  cached_model_size_ = model_->size();
  return *this;
}

bool Type::operator==(Type const& other) const {
  if (model_->getId() != other.model_->getId()) {
    return false;
  }
  return model_->equals(other.model_.get());
}

std::ostream& operator<<(std::ostream& os, const Type& type) {
  os << type.toString();
  return os;
}

std::string Type::toString() const {
  return fmt::format("{}<{}>", model_->getId(), model_->toString());
}

Type Void;
Type I8 = makePtType(PT_I8);
Type U8 = makePtType(PT_U8);
Type I16 = makePtType(PT_I16);
Type U16 = makePtType(PT_U16);
Type I32 = makePtType(PT_I32);
Type U32 = makePtType(PT_U32);
Type I64 = makePtType(PT_I64);
Type U64 = makePtType(PT_U64);
Type F16 = makePtType(PT_F16);
Type F32 = makePtType(PT_F32);
Type F64 = makePtType(PT_F64);
Type I128 = makePtType(PT_I128);
Type U128 = makePtType(PT_U128);
Type I1 = makePtType(PT_I1);
Type CF32 = makePtType(PT_CF32);
Type CF64 = makePtType(PT_CF64);

bool isFloatTy(const Type& type) {
  if (!type.isa<PtTy>()) {
    return false;
  }

  return type == F16 || type == F32 || type == F64;
}

bool isIntTy(const Type& type) {
  if (!type.isa<PtTy>()) {
    return false;
  }

  const PtType pt_type = type.as<PtTy>()->pt_type();
  return pt_type == PT_I8 || pt_type == PT_U8 || pt_type == PT_I16 ||
         pt_type == PT_U16 || pt_type == PT_I32 || pt_type == PT_U32 ||
         pt_type == PT_I64 || pt_type == PT_U64 || pt_type == PT_I128 ||
         pt_type == PT_U128;
}

Type Type::fromString(std::string_view repr) {
  // Extract keyword
  auto less = repr.find_first_of('<');
  auto keyword = repr.substr(0, less);
  auto details = repr.substr(less + 1);

  SPU_ENFORCE(!keyword.empty());
  SPU_ENFORCE(!details.empty());
  SPU_ENFORCE(details.back() == '>');

  // Remove trailing >
  details = details.substr(0, details.length());

  // Find builder function
  auto fctor = TypeContext::getTypeContext()->getTypeCreateFunction(keyword);

  return Type(fctor(details.substr(0, details.length() - 1)));
}

void PtTy::fromString(std::string_view detail) {
  auto pt_type = magic_enum::enum_cast<PtType>(detail);
  SPU_ENFORCE(pt_type.has_value(), "parse failed from={}", detail);
  pt_type_ = pt_type.value();
}

std::string PtTy::toString() const {
  return std::string(magic_enum::enum_name(pt_type_));
}

void RingTy::fromString(std::string_view detail) {
  auto field = magic_enum::enum_cast<FieldType>(detail);
  SPU_ENFORCE(field.has_value(), "parse failed from={}", detail);
  field_ = field.value();
}

std::string RingTy::toString() const {
  return std::string(magic_enum::enum_name(field()));
}

void GfmpTy::fromString(std::string_view detail) {
  auto comma = detail.find_first_of(',');
  auto field_str = detail.substr(0, comma);
  auto mp_exp_str = detail.substr(comma + 1);
  auto field = magic_enum::enum_cast<FieldType>(field_str);
  SPU_ENFORCE(field.has_value(), "parse failed from={}", detail);
  field_ = field.value();
  mersenne_prime_exp_ = std::stoul(std::string(mp_exp_str));
  prime_ = (static_cast<uint128_t>(1) << mersenne_prime_exp_) - 1;
}

std::string GfmpTy::toString() const {
  return fmt::format("{},{}", magic_enum::enum_name(field()),
                     mersenne_prime_exp_);
}

}  // namespace spu
