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

// This is the operation definition file for ring ops.

#ifndef SPU_DIALECT_RING_OPS
#define SPU_DIALECT_RING_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "libspu/dialect/ring/IR/dialect.td"
include "libspu/dialect/ring/IR/types.td"

//===----------------------------------------------------------------------===//
// Common traits
//===----------------------------------------------------------------------===//
class Ring_NativeOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::spu::ring::OpTrait";
}

def Ring_SameOperandsAndResultsSemanticRingType
    : Ring_NativeOpTrait<"SameOperandsAndResultsSemanticRingType">;

def Ring_SameOperandsSemanticRingType
    : Ring_NativeOpTrait<"SameOperandsSemanticRingType">;

//===----------------------------------------------------------------------===//
// Common op kinds
//===----------------------------------------------------------------------===//
class Ring_Op<string mnemonic, list<Trait> traits>
    : Op<Ring_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Ring elementwise op definitions.
//===----------------------------------------------------------------------===//
def Ring_EncodeToFxpOp : Ring_Op<"encode_to_fxp", [Pure, SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
    AnyFloatContainer : $operand,
    I64Attr : $fxp_bits
  );

  let results = (outs AnySignlessIntegerContainer: $result);

  let assemblyFormat = [{
    $operand `,` `fxp_bits` `=` $fxp_bits attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def Ring_DecodeFromFxpOp : Ring_Op<"decode_from_fxp", [Pure, SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
    AnySignlessIntegerContainer : $operand,
    I64Attr : $fxp_bits
  );

  let results = (outs AnyFloatContainer: $result);

  let assemblyFormat = [{
    $operand `,` `fxp_bits` `=` $fxp_bits attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def Ring_P2SOp : Ring_Op<"p2s", [Pure,Elementwise, SameOperandsAndResultShape,
                                 Ring_SameOperandsAndResultsSemanticRingType]> {
  let arguments = (ins
      AnyIntegerContainer: $operand
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def Ring_CastOp: Ring_Op<"cast", [Pure, Elementwise, SameOperandsAndResultShape]> {
  let arguments = (ins
      AnySecretContainer: $operand
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];

  let hasFolder = 1;
}

def Ring_TruncOp : Ring_Op<"trunc", [Pure, Elementwise, SameOperandsAndResultType]> {
  let arguments = (ins
      AnySecretContainer: $operand,
      I64Attr: $bits,
      DefaultValuedAttr<BoolAttr, "false">:$known_positive
  );
  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Ring_BitRevOp : Ring_Op<"bitrev", [Pure, Elementwise, SameOperandsAndResultType]> {
  let arguments = (ins
      AnySecretContainer: $operand,
      I64Attr: $start,
      I64Attr: $end
  );
  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

// Unary op
def Ring_NotOp : Ring_Op<"not", [Pure, Elementwise, SameOperandsAndResultType]> {
  let arguments = (ins
      AnySecretIntegerContainer: $operand
  );

  let results = (outs AnySecretIntegerContainer: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

// Unary op
def Ring_NegOp : Ring_Op<"neg", [Pure, Elementwise, SameOperandsAndResultType]> {
  let arguments = (ins
      AnySecretContainer: $operand
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def Ring_MsbOp : Ring_Op<"msb", [Pure, Elementwise, SameOperandsAndResultShape,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                 DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let arguments = (ins
      AnySecretContainer: $operand
  );

  let results = (outs SecretBooleanContainer: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

// Binary op
def Ring_AddOp : Ring_Op<"add", [Pure, Elementwise, Commutative,
                                 SameOperandsAndResultShape,
                                 Ring_SameOperandsAndResultsSemanticRingType,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                 DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let arguments = (ins
      AnyRingLike: $lhs,
      AnyRingLike: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def Ring_AndOp : Ring_Op<"and", [Pure, Elementwise, Commutative,
                                 SameOperandsAndResultShape,
                                 Ring_SameOperandsAndResultsSemanticRingType,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                 DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let arguments = (ins
      AnyRingLike: $lhs,
      AnyRingLike: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def Ring_XorOp : Ring_Op<"xor", [Pure, Elementwise, Commutative,
                                 SameOperandsAndResultShape,
                                 Ring_SameOperandsAndResultsSemanticRingType,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                 DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let arguments = (ins
      AnyRingLike: $lhs,
      AnyRingLike: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def Ring_MulOp   : Ring_Op<"mul", [Pure, Elementwise, Commutative, SameOperandsAndResultShape]> {
  let arguments = (ins
      AnyRingLike: $lhs,
      AnyRingLike: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];

  let hasCanonicalizer = 1;
}

def Ring_DotOp   : Ring_Op<"dot", [Pure]> {
  let arguments = (ins
      AnyRingLike: $lhs,
      AnyRingLike: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Ring_EqualOp : Ring_Op<"equal", [Pure, Elementwise, SameOperandsAndResultShape,
                                     Ring_SameOperandsSemanticRingType, Commutative,
                                     DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                     DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let arguments = (ins
      AnyRingLike: $lhs,
      AnyRingLike: $rhs
  );

  let results = (outs SecretBooleanContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Ring_LessOp : Ring_Op<"less", [Pure, Elementwise, SameOperandsAndResultShape,
                                   Ring_SameOperandsSemanticRingType,
                                   DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                   DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let arguments = (ins
      AnyRingLike: $lhs,
      AnyRingLike: $rhs
  );

  let results = (outs SecretBooleanContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

def Ring_LShiftOp  : Ring_Op<"lshift", [Pure, Elementwise,
                                        SameOperandsAndResultShape,
                                        Ring_SameOperandsAndResultsSemanticRingType,
                                        AllTypesMatch<["lhs", "result"]>]> {
  let arguments = (ins
      AnySecretContainer: $lhs,
      AnyIntegerContainer: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def Ring_RShiftOP  : Ring_Op<"rshift", [Pure, Elementwise,
                                        SameOperandsAndResultShape,
                                        Ring_SameOperandsAndResultsSemanticRingType,
                                        AllTypesMatch<["lhs", "result"]>]> {
  let arguments = (ins
      AnySecretContainer: $lhs,
      AnyIntegerContainer: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def Ring_ARShiftOp : Ring_Op<"arshift", [Pure, Elementwise,
                                         SameOperandsAndResultShape,
                                         Ring_SameOperandsAndResultsSemanticRingType,
                                         AllTypesMatch<["lhs", "result"]>]> {
  let arguments = (ins
      AnySecretContainer: $lhs,
      AnyIntegerContainer: $rhs
  );

  let results = (outs AnySecretContainer: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def Ring_SecretExtractSliceOp: Ring_Op<"secret_extract_slice", [Pure]> {
  let arguments = (ins AnyRingLike:$operand,
                       Variadic<AnySecretIntegerContainer>:$indices,
                       DenseI64ArrayAttr: $indexing_dim
  );

  let results = (outs AnySecretContainer:$result);

  let assemblyFormat = [{
    operands `indexing_dim` `=` $indexing_dim
      attr-dict `:` functional-type(operands, results)
  }];
}

def Ring_SecretInsertSliceOp: Ring_Op<"secret_insert_slice",
      [Pure, AllShapesMatch<["operand", "result"]>]> {
  let arguments = (ins AnyRingLike:$operand,
                       AnyRingLike:$update,
                       Variadic<AnyScalarIntegerLike>:$start_indices
  );
  let results = (outs AnySecretContainer:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif  // SPU_DIALECT_RING_OPS
