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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "libspu/dialect/pphlo/IR/base_enums.h"
#include "libspu/dialect/pphlo/IR/interface.h"  // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "libspu/dialect/pphlo/IR/types.h.inc"

namespace mlir::spu::pphlo {

class TypeTools {
 private:
  MLIRContext *context_;

 public:
  explicit TypeTools(MLIRContext *context) : context_(context) {
    (void)context_;
  }

  bool isFloatType(const Type &t) const;
  bool isFixedPointType(const Type &t) const;
  bool isFloatOrFixedPointType(const Type &t) const {
    return isFloatType(t) || isFixedPointType(t);
  }
  bool isIntType(const Type &t) const;
  bool isComplexFixedPointType(const Type &t) const;

  bool isSecretType(const Type &t) const;
  bool isPublicType(const Type &t) const { return !isSecretType(t); }

  int64_t getFxpBits(const Type &t) const;
  int64_t getFxpWidth(const Type &t) const;
  int64_t getIntWidth(const Type &t) const;

  int64_t getIntOrFxpWidth(const Type &t) const {
    if (isFixedPointType(t)) {
      return getFxpWidth(t);
    }
    return getIntWidth(t);
  }

  // Get type based on vis and base type
  Type getType(const Type &base, Visibility vis) const;

  Type getExpressedType(const Type &t) const {
    return getType(t, Visibility::PUBLIC);
  }

  // Get enum from a pphlo type
  Visibility getTypeVisibility(const Type &t) const;

  // Calculate common visibility
  Visibility computeCommonVisibility(llvm::ArrayRef<Visibility> vis) const;

  Type replaceBaseType(const Type &type, const Type &new_base) const;
  Type getBaseType(const Type &type) const;

  Type promoteToFloatType(const Type &t) const;
  Type demoteTypeToInt(const Type &t, bool is_unsigned = false) const;
};

}  // namespace mlir::spu::pphlo
