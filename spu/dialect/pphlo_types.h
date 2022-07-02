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

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "spu/dialect/pphlo_base_enums.h"

#define GET_TYPEDEF_CLASSES
#include "yasl/base/exception.h"

#include "spu/dialect/pphlo_types.h.inc"

namespace mlir::pphlo {

class TypeTools {
 public:
  template <class T>
  bool isMPCType(const Type &t) const {
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return isMPCType<T>(rt.getElementType());
    }
    return t.isa<T>();
  }

  template <class T>
  bool isExpressedType(const Type &t) const {
    auto et = getExpressedType(t);
    return et.isa<T>();
  }

  Type getExpressedType(const Type &t) const {
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return getExpressedType(rt.getElementType());
    }
    if (const auto &ut = t.dyn_cast<UnsetType>()) {
      return ut.getBase();
    }
    if (const auto &pt = t.dyn_cast<PublicType>()) {
      return pt.getBase();
    }
    if (const auto &st = t.dyn_cast<SecretType>()) {
      return st.getBase();
    }
    return t;  // Not MPC type
  }

  template <class T>
  Type toMPCType(const Type &t) const {
    if (isMPCType<T>(t)) {
      return t;
    }
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(rt.getShape(),
                                   toMPCType<T>(rt.getElementType()));
    }
    return T::get(getExpressedType(t));
  }

  Visibility getTypeVisibility(const Type &t) const {
    if (isMPCType<PublicType>(t)) {
      return Visibility::VIS_PUBLIC;
    }
    YASL_ENFORCE(isMPCType<SecretType>(t));
    return Visibility::VIS_SECRET;
  }

  static Visibility inferResultVisibility(
      llvm::ArrayRef<Visibility> input_vis) {
    if (llvm::any_of(input_vis, [](Visibility v) {
          return v == Visibility::VIS_SECRET;
        })) {
      return Visibility::VIS_SECRET;
    }
    return Visibility::VIS_PUBLIC;
  }

  Type getTypeWithVisibility(Type type, Visibility vis) const {
    switch (vis) {
      case Visibility::VIS_PUBLIC:
        return toMPCType<PublicType>(type);
      case Visibility::VIS_SECRET:
        return toMPCType<SecretType>(type);
    }
    llvm_unreachable("Should not reach here.");
  }
};

}  // namespace mlir::pphlo
