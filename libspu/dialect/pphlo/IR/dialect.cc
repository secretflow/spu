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

#include "libspu/dialect/pphlo/IR/dialect.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#include "libspu/dialect/pphlo/IR/attrs.h"  // IWYU pragma: keep
#include "libspu/dialect/pphlo/IR/ops.h"    // IWYU pragma: keep
#include "libspu/dialect/pphlo/IR/types.h"  // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "libspu/dialect/pphlo/IR/attrs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "libspu/dialect/pphlo/IR/dialect.cc.inc"
#include "libspu/dialect/pphlo/IR/types.cc.inc"

namespace mlir::spu::pphlo {

void PPHloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libspu/dialect/pphlo/IR/ops.cc.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "libspu/dialect/pphlo/IR/types.cc.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "libspu/dialect/pphlo/IR/attrs.cc.inc"
      >();
}

Type PPHloDialect::parseType(DialectAsmParser& parser) const {
  StringRef mnemonic;
  Type parsedType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, parsedType);
  if (parseResult.has_value()) {
    return parsedType;
  }
  parser.emitError(parser.getNameLoc()) << "unknown pphlo type: " << mnemonic;
  return nullptr;
}

void PPHloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (succeeded(generatedTypePrinter(type, os))) {
    return;
  }
  os << "<unknown pphlo type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute PPHloDialect::parseAttribute(DialectAsmParser& parser,
                                       Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) {
    return attr;
  }
  parser.emitError(parser.getNameLoc(), "unknown mhlo attribute");
  return {};
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void PPHloDialect::printAttribute(Attribute attr, DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the value generated to represent the constant value.
/// Otherwise, it should return nullptr on failure.
Operation* PPHloDialect::materializeConstant(OpBuilder& builder,
                                             Attribute value, Type type,
                                             Location loc) {
  auto op =
      builder.create<arith::ConstantOp>(loc, dyn_cast<ElementsAttr>(value));
  if (op.getType() != type) {
    return nullptr;
  }
  return op;
}

}  // namespace mlir::spu::pphlo
