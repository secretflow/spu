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

include "libspu/dialect/pphlo/IR/dialect.td"
include "libspu/dialect/pphlo/IR/interface.td"
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
  let summary = "A secret type";

  let parameters = (ins "Type":$baseType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$baseType), [{
      return $_get(baseType.getContext(), baseType);
    }]>
  ];
  let assemblyFormat = "`<` $baseType `>`";
}

def PPHLO_FixedPointType : PPHLO_BaseType<"FixedPoint", "fxp"> {
  let summary = "A fixed-point type";

  let parameters = (ins "unsigned": $width,
                        "unsigned": $fraction
  );

  let assemblyFormat = "`<`$width`,`$fraction`>`";
}

def PPHLO_ComplexFixedPointType : PPHLO_BaseType<"CplxFxp", "cplxfxp"> {
  let summary = "A complex fixed-point type";

  let parameters = (ins "unsigned": $width,
                        "unsigned": $fraction
  );

  let assemblyFormat = "`<`$width`,`$fraction`>`";
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

def PPHLO_PublicInt : SignlessIntOfWidths<[8, 16, 32, 64, 128]>;
def PPHLO_SecretInt : PPHlo_SecretTypeOf<[I8, I16, I32, I64, I128]>;

def PPHLO_PublicUInt : UnsignedIntOfWidths<[8, 16, 32, 64]>;
def PPHLO_SecretUInt : PPHlo_SecretTypeOf<[PPHLO_PublicUInt]>;

def PPHLO_PublicFloat : AnyTypeOf<[F16, F32, F64]>;
def PPHLO_SecretFloat : PPHlo_SecretTypeOf<[PPHLO_PublicFloat]>;

def PPHLO_PublicFixedPoint : AnyTypeOf<[PPHLO_FixedPointType]>;
def PPHLO_SecretFixedPoint : PPHlo_SecretTypeOf<[PPHLO_FixedPointType]>;

def PPHLO_PublicComplex : Complex<AnyTypeOf<[F32, F64]>>;
def PPHLO_SecretComplex : PPHlo_SecretTypeOf<[PPHLO_PublicComplex]>;

def PPHLO_PublicComplexFixedPoint : AnyTypeOf<[PPHLO_ComplexFixedPointType]>;
def PPHLO_SecretComplexFixedPoint : PPHlo_SecretTypeOf<[PPHLO_ComplexFixedPointType]>;

def PPHLO_BOOL : AnyTypeOf<[PPHLO_PublicBool, PPHLO_SecretBool]>;
def PPHLO_SInt : AnyTypeOf<[PPHLO_PublicInt, PPHLO_SecretInt]>;
def PPHLO_Int : AnyTypeOf<[PPHLO_PublicInt, PPHLO_SecretInt, PPHLO_PublicUInt, PPHLO_SecretUInt]>;
def PPHLO_Float : AnyTypeOf<[PPHLO_PublicFloat, PPHLO_SecretFloat]>;
def PPHLO_FixedPoint : AnyTypeOf<[PPHLO_PublicFixedPoint, PPHLO_SecretFixedPoint]>;
def PPHLO_Complex : AnyTypeOf<[PPHLO_PublicComplex, PPHLO_SecretComplex]>;
def PPHLO_ComplexFixedPoint : AnyTypeOf<[PPHLO_PublicComplexFixedPoint, PPHLO_SecretComplexFixedPoint]>;

//===----------------------------------------------------------------------===//
// PPHLO Tensor Types
//===----------------------------------------------------------------------===//

// ----- Simple ------- //
def PPHLO_BoolTensor : RankedTensorOf<[PPHLO_BOOL]>;
def PPHLO_SIntTensor : RankedTensorOf<[PPHLO_SInt]>;
def PPHLO_IntTensor : RankedTensorOf<[PPHLO_Int]>;
def PPHLO_ScalarIntTensor : 0DTensorOf<[PPHLO_Int]>;
def PPHLO_FpTensor : RankedTensorOf<[PPHLO_Float]>;
def PPHLO_FxpTensor : RankedTensorOf<[PPHLO_FixedPoint]>;
def PPHLO_FpComplexTensor : RankedTensorOf<[PPHLO_Complex]>;
def PPHLO_FxpComplexTensor : RankedTensorOf<[PPHLO_ComplexFixedPoint]>;
// ----- Combine ------- //
def PPHLO_SignedIntOrFpOrFxp : AnyTypeOf<[PPHLO_SIntTensor, PPHLO_FpTensor, PPHLO_FxpTensor]>;
def PPHLO_BoolOrIntTensor : AnyTypeOf<[PPHLO_BoolTensor, PPHLO_IntTensor]>;
def PPHLO_IntOrFpTensor : AnyTypeOf<[PPHLO_IntTensor, PPHLO_FpTensor]>;
def PPHLO_IntOrFxpTensor : AnyTypeOf<[PPHLO_IntTensor, PPHLO_FxpTensor]>;
def PPHLO_FpOrComplexTensor :  AnyTypeOf<[PPHLO_FpTensor, PPHLO_FpComplexTensor]>;
def PPHLO_FpOrFxpTensor: AnyTypeOf<[PPHLO_FpTensor, PPHLO_FxpTensor]>;
def PPHLO_FxpOrComplexTensor : AnyTypeOf<[PPHLO_FxpTensor, PPHLO_FxpComplexTensor]>;
def PPHLO_FpOrFxpOrComplexTensor: AnyTypeOf<[PPHLO_FpOrComplexTensor, PPHLO_FxpOrComplexTensor]>;
def PPHLO_AllComplexTensor: AnyTypeOf<[PPHLO_FpComplexTensor, PPHLO_FxpComplexTensor]>;
def PPHLO_NoFxpTensor : RankedTensorOf<[PPHLO_BOOL, PPHLO_Int, PPHLO_Float]>;
def PPHLO_NoComplexTensor : RankedTensorOf<[PPHLO_BOOL, PPHLO_Int, PPHLO_Float, PPHLO_FixedPoint]>;
def PPHLO_Tensor : RankedTensorOf<[PPHLO_BOOL, PPHLO_Int, PPHLO_Float, PPHLO_FixedPoint, PPHLO_Complex, PPHLO_ComplexFixedPoint]>;
// ------ Any ---------- //
def PPHLO_AnyTensor: RankedTensorOf<[AnyType]>;

#endif  // SPU_DIALECT_PPHLO_TYPES
