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

#include "libspu/dialect/ring/IR/dialect.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep

#include "libspu/dialect/ring/IR/bytecode.h"
// #include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
// #include "mlir/IR/OpDefinition.h"

#include "libspu/dialect/ring/IR/ops.h"    // IWYU pragma: keep
#include "libspu/dialect/ring/IR/types.h"  // IWYU pragma: keep
#include "libspu/version.h"

#define GET_TYPEDEF_CLASSES
#include "libspu/dialect/ring/IR/dialect.cc.inc"
#include "libspu/dialect/ring/IR/types.cc.inc"

namespace mlir::spu::ring {

void RingDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libspu/dialect/ring/IR/ops.cc.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "libspu/dialect/ring/IR/types.cc.inc"
      >();

  RingDialectVersion version(::spu::getVersionStr());
  if (version.isValid()) {
    setVersion(version);
  }

  addBytecodeInterface(this);
}

Type RingDialect::parseType(DialectAsmParser& parser) const {
  StringRef mnemonic;
  Type parsedType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, parsedType);
  if (parseResult.has_value()) {
    return parsedType;
  }
  parser.emitError(parser.getNameLoc()) << "unknown ring type: " << mnemonic;
  return nullptr;
}

void RingDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (succeeded(generatedTypePrinter(type, os))) {
    return;
  }
  os << "<unknown ring type>";
}

/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the value generated to represent the constant value.
/// Otherwise, it should return nullptr on failure.
Operation* RingDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<arith::ConstantOp>(loc, type,
                                           dyn_cast<ElementsAttr>(value));
}

}  // namespace mlir::spu::ring
