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
//

//===----------------------------------------------------------------------===//
//
// Defines the MLIR Ring dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_DIALECT_RING_DIALECT
#define SPU_DIALECT_RING_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Ring Dialect
//===----------------------------------------------------------------------===//

def Ring_Dialect : Dialect {
  string summary = "Ring dialect";
  string description = [{
  }];
  let name = "ring";
  let cppNamespace = "::mlir::spu::ring";
  let useDefaultTypePrinterParser = 0;
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &os) const override;

    std::optional<RingDialectVersion> getVersion() const { return version; }
    void setVersion(std::optional<RingDialectVersion> version) { this->version = version; };

  private:
    std::optional<RingDialectVersion> version;
  }];
}

#endif  // SPU_DIALECT_RING_DIALECT
