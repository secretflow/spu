// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/device/utils/utils.h"

#include "libspu/core/prelude.h"
#include "libspu/dialect/ring/IR/type_helper.h"
#include "libspu/dialect/utils/utils.h"

namespace spu::device {

std::string defaultOpNamePrinter(mlir::Operation &op) {
  return op.getName().getStringRef().str();
}

spu::SemanticType getSemanticTypeFromMlirType(mlir::Type mlir_ty) {
  auto baseType = mlir::spu::ring::getBaseType(mlir_ty);

  if (mlir::isa<mlir::IndexType>(baseType)) {
    return SE_I64;
  }

  if (auto it = mlir::dyn_cast<mlir::IntegerType>(baseType)) {
    if (it.getWidth() == 1) {
      return spu::SE_1;
    }
    // In mlir, isSigned is for si[1-9][0-9]* type, isUnsigned is for
    // ui[1-9][0-9]*, i[1-9][0-9]* is signless IntegerType... So here, we only
    // check for isUnsigned, signless we treat it as signed.
    // See https://reviews.llvm.org/D72533
    switch (it.getWidth()) {
      case 8:
        return it.isUnsigned() ? spu::SE_U8 : spu::SE_I8;
      case 16:
        return it.isUnsigned() ? spu::SE_U16 : spu::SE_I16;
      case 32:
        return it.isUnsigned() ? spu::SE_U32 : spu::SE_I32;
      case 64:
        return it.isUnsigned() ? spu::SE_U64 : spu::SE_I64;
      case 128:
        return spu::SE_I128;
    }
  } else if (auto ft = mlir::dyn_cast<mlir::FloatType>(baseType)) {
    switch (ft.getWidth()) {
      case 16:
        return SE_I16;
      case 32:
        return SE_I32;
      case 64:
        return SE_I64;
    }
  }
  SPU_THROW("invalid type {}", mlir::spu::mlirObjectToString(mlir_ty));
}

spu::PtType getPtTypeFromMlirType(mlir::Type mlir_ty) {
  auto baseType = mlir::getElementTypeOrSelf(mlir_ty);
  if (auto ft = mlir::dyn_cast<mlir::FloatType>(baseType)) {
    switch (ft.getWidth()) {
      case 16:
        return spu::PT_F16;
      case 32:
        return spu::PT_F32;
      case 64:
        return spu::PT_F64;
    }
  } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(baseType)) {
    if (it.getWidth() == 1) {
      return spu::PT_I1;
    }
    // In mlir, isSigned is for si[1-9][0-9]* type, isUnsigned is for
    // ui[1-9][0-9]*, i[1-9][0-9]* is signless IntegerType... So here, we only
    // check for isUnsigned, signless we treat it as signed.
    // See https://reviews.llvm.org/D72533
    switch (it.getWidth()) {
      case 8:
        return it.isUnsigned() ? spu::PT_U8 : spu::PT_I8;
      case 16:
        return it.isUnsigned() ? spu::PT_U16 : spu::PT_I16;
      case 32:
        return it.isUnsigned() ? spu::PT_U32 : spu::PT_I32;
      case 64:
        return it.isUnsigned() ? spu::PT_U64 : spu::PT_I64;
      case 128:
        return spu::PT_I128;
    }
  } else if (mlir::isa<mlir::IndexType>(baseType)) {
    return spu::PT_I64;
  }

  SPU_THROW("invalid type {}", mlir::spu::mlirObjectToString(mlir_ty));
}

}  // namespace spu::device
