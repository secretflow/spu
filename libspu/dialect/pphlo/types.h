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

#include "libspu/dialect/pphlo/base_enums.h"
#include "libspu/dialect/pphlo/interface.h"  // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "libspu/dialect/pphlo/types.h.inc"

namespace mlir::spu::pphlo {

class TypeTools {
 private:
  [[maybe_unused]] MLIRContext *context_;

 public:
  explicit TypeTools(MLIRContext *context) : context_(context) {}

  bool isFloatType(const Type &t) const;
  bool isIntType(const Type &t) const;

  bool isSecretType(const Type &t) const;

  // Get type based on vis and base type
  Type getType(const Type &base, Visibility vis) const;

  Type getExpressedType(const Type &t) const {
    return getType(t, Visibility::PUBLIC);
  }

  // Get enum from a pphlo type
  Visibility getTypeVisibility(const Type &t) const;

  // Calculate common visibility
  Visibility computeCommonVisibility(llvm::ArrayRef<Visibility> vis) const;
};

}  // namespace mlir::spu::pphlo
