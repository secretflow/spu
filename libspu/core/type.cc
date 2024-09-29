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

namespace spu {

Type::Type() : impl_(std::make_unique<VoidTy>()) {}

Type::Type(std::unique_ptr<TypeObject> impl) : impl_(std::move(impl)) {}

Type& Type::operator=(const Type& other) {
  if (this != &other) {
    impl_ = other.impl_->clone();
  }
  return *this;
}

bool Type::operator==(Type const& other) const {
  if (impl_->getId() != other.impl_->getId()) {
    return false;
  }
  return impl_->equals(other.impl_.get());
}

std::ostream& operator<<(std::ostream& os, const Type& type) {
  os << type.toString();
  return os;
}

std::string Type::toString() const {
  return fmt::format("{}<{}>", impl_->getId(), impl_->toString());
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

}  // namespace spu
