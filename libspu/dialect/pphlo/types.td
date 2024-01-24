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

include "dialect.td"
include "interface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// PPHLO Types Classes
//===----------------------------------------------------------------------===//
class PPHLO_BaseType<string name, string typeMnemonic>
    : TypeDef<PPHlo_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def PPHLO_SecretType : PPHLO_BaseType<"Secret", "secret"> {
  let summary = "A secret value";

  let parameters = (ins "Type":$baseType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$baseType), [{
      return $_get(baseType.getContext(), baseType);
    }]>
  ];
  let assemblyFormat = "`<` $baseType `>`";
}

def IsSecretTypePred :
        CPred<"::llvm::isa<::mlir::spu::pphlo::SecretType>($_self)">;

class PPHlo_SecretTypeOf<list<Type> allowedTypes,
                          string cppClassName = "::mlir::spu::pphlo"> :
    Type<And<[IsSecretTypePred, Concat<"[](::mlir::Type baseType) { return ",
                SubstLeaves<"$_self", "baseType",
                AnyTypeOf<allowedTypes>.predicate>,
                "; }(::llvm::cast<::mlir::spu::pphlo::SecretType>($_self).getBaseType())">]>,
         "Secret of " # AnyTypeOf<allowedTypes>.summary # " values", cppClassName>;

//===----------------------------------------------------------------------===//
// PPHLO Type Groups
//===----------------------------------------------------------------------===//
def PPHLO_PublicBool : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;
def PPHLO_SecretBool : PPHlo_SecretTypeOf<[I1]>;

def PPHLO_PublicInt : SignlessIntOfWidths<[8, 16, 32, 64]>;
def PPHLO_SecretInt : PPHlo_SecretTypeOf<[I8, I16, I32, I64]>;

def PPHLO_PublicUInt : UnsignedIntOfWidths<[8, 16, 32, 64]>;
def PPHLO_SecretUInt : PPHlo_SecretTypeOf<[PPHLO_PublicUInt]>;

def PPHLO_PublicFloat : AnyTypeOf<[F16, F32, F64]>;
def PPHLO_SecretFloat : PPHlo_SecretTypeOf<[PPHLO_PublicFloat]>;

def PPHLO_PublicComplex : Complex<AnyTypeOf<[F32, F64]>>;
def PPHLO_SecretComplex : PPHlo_SecretTypeOf<[PPHLO_PublicComplex]>;

def PPHLO_BOOL : AnyTypeOf<[PPHLO_PublicBool, PPHLO_SecretBool]>;
def PPHLO_Int : AnyTypeOf<[PPHLO_PublicInt, PPHLO_SecretInt, PPHLO_PublicUInt, PPHLO_SecretUInt]>;
def PPHLO_Float : AnyTypeOf<[PPHLO_PublicFloat, PPHLO_SecretFloat]>;
def PPHLO_Complex : AnyTypeOf<[PPHLO_PublicComplex, PPHLO_SecretComplex]>;

//===----------------------------------------------------------------------===//
// PPHLO Tensor Types
//===----------------------------------------------------------------------===//
def PPHLO_Tensor : StaticShapeTensorOf<[PPHLO_BOOL, PPHLO_Int, PPHLO_Float, PPHLO_Complex]>;
def PPHLO_BoolTensor : StaticShapeTensorOf<[PPHLO_BOOL]>;
def PPHLO_IntTensor : StaticShapeTensorOf<[PPHLO_Int]>;
def PPHLO_BoolOrIntTensor : StaticShapeTensorOf<[PPHLO_Int, PPHLO_BOOL]>;
def PPHLO_FpTensor : StaticShapeTensorOf<[PPHLO_Float]>;
def PPHLO_ScalarIntTensor : 0DTensorOf<[PPHLO_Int]>;
def PPHLO_ComplexTensor :  StaticShapeTensorOf<[PPHLO_Complex]>;
def PPHLO_FpOrComplexTensor :  StaticShapeTensorOf<[PPHLO_Complex, PPHLO_Float]>;
def PPHLO_AnyTensor: StaticShapeTensorOf<[AnyType]>;

#endif  // SPU_DIALECT_PPHLO_TYPES
