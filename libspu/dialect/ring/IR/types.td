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
// This file declares the Ring dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_DIALECT_RING_TYPES
#define SPU_DIALECT_RING_TYPES

include "libspu/dialect/ring/IR/dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Ring Types Classes
//===----------------------------------------------------------------------===//
class BaseSecretType<string name, string typeMnemonic>
    : TypeDef<Ring_Dialect, name> {
  let mnemonic = typeMnemonic;
  let parameters = (ins "Type":$baseType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$baseType), [{
      return $_get(baseType.getContext(), baseType);
    }]>
  ];
  let assemblyFormat = "`<` $baseType `>`";
}

def SecretType   : BaseSecretType<"Secret", "secret">;

def IsSecretType : CPred<
  "::llvm::isa<::mlir::spu::ring::SecretType>($_self)"
>;

class SecretTypeOf<list<Type> allowedTypes, string summary,
                   string cppClassName = "::mlir::spu::ring"> :
    Type<And<[IsSecretType, Concat<"[](::mlir::Type baseType) { return ",
                SubstLeaves<"$_self", "baseType",
                AnyTypeOf<allowedTypes>.predicate>,
                "; }(::llvm::cast<::mlir::spu::ring::SecretType>($_self).getBaseType())">]>,
         "Secret of " # summary # " values", cppClassName>;

//===----------------------------------------------------------------------===//
// Types Classes
//===----------------------------------------------------------------------===//
def AnySignlessIntegerContainer : ValueSemanticsContainerOf<[AnySignlessInteger]>;
def AnyIntegerContainer: ValueSemanticsContainerOf<[AnySignlessInteger, AnyUnsignedInteger]>;
def AnyFloatContainer : ValueSemanticsContainerOf<[AnyFloat]>;

// Secret scalar
def SecretBoolean         : SecretTypeOf<[I1], "secret boolean">;
def SecretSignlessInteger : SecretTypeOf<[AnySignlessInteger], "secret signless integer">;
def SecretUnsignedInteger : SecretTypeOf<[AnyUnsignedInteger], "secret unsigned integer">;

// Secret container like
def SecretBooleanContainer         : ValueSemanticsContainerOf<[SecretBoolean]>;
def SecretSignlessIntegerContainer : ValueSemanticsContainerOf<[SecretSignlessInteger]>;
def SecretUnsignedIntegerContainer : ValueSemanticsContainerOf<[SecretUnsignedInteger]>;

// Aggregate
def AnyScalarIntegerLike      : AnyTypeOf<[AnySignlessInteger, AnyUnsignedInteger,
                                           SecretSignlessInteger, SecretUnsignedInteger], "scalar integer like">;
def AnySecretIntegerContainer : AnyTypeOf<[SecretSignlessIntegerContainer, SecretUnsignedIntegerContainer], "any-secret-integer-container">;
def AnySecretContainer        : AnyTypeOf<[SecretBooleanContainer, SecretSignlessIntegerContainer, SecretUnsignedIntegerContainer], "any-secret-container">;

def AnyRingLike : AnyTypeOf<[ValueSemanticsContainerOf<[AnySignlessInteger]>,
                             ValueSemanticsContainerOf<[AnyUnsignedInteger]>,
                             AnySecretContainer], "any-ring-like-container">;

#endif  // SPU_DIALECT_RING_TYPES
