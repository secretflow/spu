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
//

//===----------------------------------------------------------------------===//
//
// Defines the MLIR Privacy-Preserving HLO(PPHLO) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_DIALECT_PPHLO_DIALECT
#define SPU_DIALECT_PPHLO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PPHLO Dialect
//===----------------------------------------------------------------------===//

def PPHlo_Dialect : Dialect {
  string summary = "Privacy-Preserving HLO(PPHLO) dialect";
  string description = [{
    PPHLO represents a high level abstraction for language use by SPU.
    It implements a subset of mlir mhlo ops with it's own privacy-preserving focused type system.

    Learn more about mlir hlo at https://github.com/tensorflow/mlir-hlo
  }];
  let name = "pphlo";
  let cppNamespace = "::mlir::pphlo";
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let extraClassDeclaration = [{
    Attribute parseAttribute(DialectAsmParser & parser, Type type)
        const override;
    void printAttribute(Attribute, DialectAsmPrinter &) const override;
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &os) const override;
  }];
}

#endif  // SPU_DIALECT_PPHLO_DIALECT
