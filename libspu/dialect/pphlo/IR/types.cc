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

#include "libspu/dialect/pphlo/IR/types.h"

#include "mlir/IR/TypeUtilities.h"

#include "libspu/core/prelude.h"

namespace mlir::spu::pphlo {

namespace utils {

// A tiny utility to handle aggregate type like tensor
template <typename Fn, typename... Args>
bool StripAllContainerType(const Type &t, Fn foo, Args &&...args) {
  if (const auto &rt = mlir::dyn_cast<RankedTensorType>(t)) {
    return StripAllContainerType(rt.getElementType(), foo,
                                 std::forward<Args>(args)...);
  }

  if (const auto &ct = mlir::dyn_cast<ComplexType>(t)) {
    return StripAllContainerType(ct.getElementType(), foo,
                                 std::forward<Args>(args)...);
  }

  if (const auto &st = mlir::dyn_cast<SecretType>(t)) {
    return StripAllContainerType(st.getBaseType(), foo,
                                 std::forward<Args>(args)...);
  }

  return foo(t, std::forward<Args>(args)...);
}

}  // namespace utils

bool TypeTools::isSecretType(const Type &t) const {
  return mlir::isa<SecretType>(getElementTypeOrSelf(t));
}

bool TypeTools::isFloatType(const Type &t) const {
  return utils::StripAllContainerType(
      t, [](const Type &t) { return mlir::isa<FloatType>(t); });
}

bool TypeTools::isIntType(const Type &t) const {
  return utils::StripAllContainerType(
      t, [](const Type &t) { return mlir::isa<IntegerType>(t); });
}

Type TypeTools::getType(const Type &type, Visibility vis) const {
  if (getTypeVisibility(type) == vis) {
    return type;
  }

  if (const auto &rt = mlir::dyn_cast<RankedTensorType>(type)) {
    return RankedTensorType::get(rt.getShape(),
                                 getType(rt.getElementType(), vis));
  }

  if (vis == Visibility::PUBLIC) {
    if (isSecretType(type)) {
      return dyn_cast<SecretType>(type).getBaseType();
    }
    return type;
  }

  SPU_ENFORCE(vis == Visibility::SECRET);

  return SecretType::get(type);
}

Visibility TypeTools::getTypeVisibility(const Type &t) const {
  if (isSecretType(t)) {
    return Visibility::SECRET;
  }
  return Visibility::PUBLIC;
}

Visibility TypeTools::computeCommonVisibility(
    llvm::ArrayRef<Visibility> vis) const {
  if (llvm::any_of(vis, [](Visibility v) { return v == Visibility::SECRET; })) {
    return Visibility::SECRET;
  }
  return Visibility::PUBLIC;
}

Type TypeTools::getBaseType(const Type &type) const {
  Type element_type;

  (void)utils::StripAllContainerType(type, [&element_type](const Type &t) {
    element_type = t;
    return true;
  });

  return element_type;
}

}  // namespace mlir::spu::pphlo
