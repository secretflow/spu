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

#include "spu/dialect/pphlo_dialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "spu/dialect/pphlo_attrs.h"
#include "spu/dialect/pphlo_ops.h"
#include "spu/dialect/pphlo_types.h"

#define GET_ATTRDEF_CLASSES
#include "spu/dialect/pphlo_attrs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "spu/dialect/pphlo_dialect.cc.inc"
#include "spu/dialect/pphlo_types.cc.inc"

namespace mlir::pphlo {

void PPHloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "spu/dialect/pphlo_ops.cc.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "spu/dialect/pphlo_types.cc.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "spu/dialect/pphlo_attrs.cc.inc"
      >();

  allowUnknownTypes(true);
  getContext()->loadDialect<tensor::TensorDialect>();
}

}  // namespace mlir::pphlo
