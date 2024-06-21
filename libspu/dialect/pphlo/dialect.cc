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

#include "libspu/dialect/pphlo/dialect.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#include "libspu/dialect/pphlo/attrs.h"  // IWYU pragma: keep
#include "libspu/dialect/pphlo/ops.h"    // IWYU pragma: keep
#include "libspu/dialect/pphlo/types.h"  // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "libspu/dialect/pphlo/attrs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "libspu/dialect/pphlo/dialect.cc.inc"
#include "libspu/dialect/pphlo/types.cc.inc"

namespace mlir::spu::pphlo {

void PPHloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libspu/dialect/pphlo/ops.cc.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "libspu/dialect/pphlo/types.cc.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "libspu/dialect/pphlo/attrs.cc.inc"
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

}  // namespace mlir::spu::pphlo
