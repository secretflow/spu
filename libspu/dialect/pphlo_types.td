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
// This file declares the Privacy-Preserving HLO dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_DIALECT_PPHLO_TYPES
#define SPU_DIALECT_PPHLO_TYPES

include "pphlo_dialect.td"
include "pphlo_base_enums.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// PPHLO Types
//===----------------------------------------------------------------------===//
class PPHLO_BaseType<string name, string typeMnemonic>
    : TypeDef<PPHlo_Dialect, name> {
  let mnemonic = typeMnemonic;
  let parameters = (ins "Type": $base);
  let builders = [
      TypeBuilderWithInferredContext<(ins "Type":$base), [{
      return $_get(base.getContext(), base);
      }]>
  ];
  let assemblyFormat = "`<` $base `>`";
}

def PPHLO_UnsetType : PPHLO_BaseType<"Unset", "unst"> {
    let summary = "PPHlo unset security type";
}

def PPHLO_PublicType : PPHLO_BaseType<"Public", "pub"> {
    let summary = "PPHlo public type";
}

def PPHLO_SecretType : PPHLO_BaseType<"Secret", "sec"> {
    let summary = "PPHlo secret type";
}

def PPHLO_PublicIntType : Type<And<[PPHLO_PublicType.predicate,
    CPred<"$_self.cast<::mlir::pphlo::PublicType>().getBase().isa<::mlir::IntegerType>()">]>, "public integer type", "::pphlo::PubIntType">;

def PPHLO_SecretIntType : Type<And<[PPHLO_SecretType.predicate,
    CPred<"$_self.cast<::mlir::pphlo::SecretType>().getBase().isa<::mlir::IntegerType>()">]>, "secret integer type", "::pphlo::SecIntType">;

def PPHLO_PublicFpType : Type<And<[PPHLO_PublicType.predicate,
    CPred<"$_self.cast<::mlir::pphlo::PublicType>().getBase().isa<::mlir::FloatType>()">]>, "public fixed-point type", "::pphlo::PubFpType">;

def PPHLO_SecretFpType : Type<And<[PPHLO_SecretType.predicate,
    CPred<"$_self.cast<::mlir::pphlo::SecretType>().getBase().isa<::mlir::FloatType>()">]>, "secret fixed-point type", "::pphlo::SecFpType">;

def PPHLO_Tensor : StaticShapeTensorOf<[PPHLO_PublicType, PPHLO_SecretType]>;
def PPHLO_IntTensor : StaticShapeTensorOf<[PPHLO_PublicIntType, PPHLO_SecretIntType]>;
def PPHLO_FpTensor : StaticShapeTensorOf<[PPHLO_PublicFpType, PPHLO_SecretFpType]>;
def PPHLO_ScalarIntTensor : 0DTensorOf<[PPHLO_PublicIntType, PPHLO_SecretIntType]>;

#endif  // SPU_DIALECT_PPHLO_TYPES
