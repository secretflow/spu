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

#pragma once

// #include "mlir/IR/Attributes.h"
// #include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
// #include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

#include "libspu/dialect/utils/assembly_format.h"

namespace mlir::spu::ring {

template <class... OpTypes>
void printSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                    OpTypes... types) {
  mlir::spu::printSameOperandsAndResultType(p, op,
                                            std::forward<OpTypes>(types)...);
}

template <class... OpTypes>
ParseResult parseSameOperandsAndResultType(OpAsmParser& parser,
                                           OpTypes&... types) {
  return mlir::spu::parseSameOperandsAndResultType(
      parser, std::forward<OpTypes&>(types)...);
}

// CustomCall target attr
inline void printCustomCallTarget(AsmPrinter& p, Operation*,
                                  StringAttr target) {
  mlir::spu::printCustomCallTargetImpl(p, target);
}

inline ParseResult parseCustomCallTarget(AsmParser& parser,
                                         StringAttr& target) {
  return mlir::spu::parseCustomCallTargetImpl(parser, target);
}

}  // namespace mlir::spu::ring