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

// This is the operation definition file for pphlo ops.

#ifndef SPU_DIALECT_PPHLO_OPS
#define SPU_DIALECT_PPHLO_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "libspu/dialect/pphlo/IR/attrs.td"
include "libspu/dialect/pphlo/IR/base_enums.td"
include "libspu/dialect/pphlo/IR/types.td"

//===----------------------------------------------------------------------===//
// Common traits
//===----------------------------------------------------------------------===//

class PPHLO_NativeOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::spu::pphlo::OpTrait";
}

def PPHLO_PairwiseSameOperandAndResultType
    : PPHLO_NativeOpTrait<"PairwiseSameOperandAndResultType">;

class PPHLO_Op<string mnemonic, list<Trait> traits>
    : Op<PPHlo_Dialect, mnemonic, traits> {
}

class PPHLO_WithShapeInferOp<string mnemonic, list<Trait> traits> :
    PPHLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
}

//===----------------------------------------------------------------------===//
// Common op kinds
//===----------------------------------------------------------------------===//
class PPHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
                               Type OperandType, Type ResultType = OperandType>
    : PPHLO_Op<mnemonic, traits #[Pure, Elementwise, SameOperandsAndResultShape]> {
  let arguments = (ins OperandType: $operand);
  let results = (outs ResultType: $result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

class PPHLO_UnaryElementwiseOpWithTypeInfer<string mnemonic, list<Trait> traits,
                               Type OperandType, Type ResultType = OperandType>
    : PPHLO_UnaryElementwiseOp<mnemonic,
                               traits # [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                         DeclareOpInterfaceMethods<InferShapedTypeOpInterface>],
                  OperandType, ResultType> {
}

class PPHLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits,
                                Type OperandType, Type ResultType = OperandType>
    : PPHLO_Op<mnemonic, traits #[Pure, SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
      OperandType: $lhs,
      OperandType: $rhs
  );

  let results = (outs ResultType: $result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

class PPHLO_BinaryElementwiseOpWithTypeInfer<string mnemonic, list<Trait> traits,
                                Type OperandType, Type ResultType = OperandType>
    : PPHLO_BinaryElementwiseOp<mnemonic,
                                traits # [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                          DeclareOpInterfaceMethods<InferShapedTypeOpInterface>],
               OperandType, ResultType> {
}

class PPHLO_BinaryLogicalElementwiseOp<string mnemonic>
    : PPHLO_BinaryElementwiseOpWithTypeInfer<
          mnemonic, [Commutative, Pure, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>], PPHLO_BoolOrIntTensor> {
}

class PPHLO_ComparisionOp<string mnemonic, list<Trait> traits = []>
    : PPHLO_BinaryElementwiseOpWithTypeInfer<
          mnemonic, traits # [Pure], PPHLO_Tensor, PPHLO_BoolTensor> {
}

//===----------------------------------------------------------------------===//
// pphlo unary elementwise op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_AbsOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"abs", [SameOperandsAndResultType], PPHLO_SignedIntOrFpOrFxp> {
  let summary = "Absolute value operator";
  let description = [{
    Performs element-wise abs operation on operand tensor and produces a result tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs
  }];
}

def PPHLO_CeilOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"ceil", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Ceil operator";
  let description = [{
    Performs element-wise ceil of operand tensor and produces a result tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#ceil
  }];
}

def PPHLO_ConvertOp
    : PPHLO_UnaryElementwiseOp<"convert", [Pure,
                                           DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]
                                        , PPHLO_NoComplexTensor> {
  let summary = "Convert operator";
  let description = [{
    Performs an element-wise conversion from one element type to another on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PPHLO_CosineOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"cosine", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "cosine operator";
  let description = [{
    Performs element-wise cosine operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cosine
  }];
}

def PPHLO_ExpOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"exponential", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Exponential operator";
  let description = [{
    Performs element-wise exponential operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential
  }];
}

def PPHLO_Expm1Op
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"exponential_minus_one", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Exponential minus one operator";
  let description = [{
    Performs element-wise exponential minus one operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one
  }];
}

def PPHLO_FloorOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"floor", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Floor operator";
  let description = [{
    Performs element-wise floor of operand tensor and produces a result tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#floor
  }];
}

def PPHLO_LogOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"log", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Log operator";
  let description = [{
    Performs element-wise logarithm operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log
  }];
}

def PPHLO_Log1pOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"log_plus_one", [SameOperandsAndResultType],
                                             PPHLO_FpOrFxpTensor> {
  let summary = "Log1p operator";
  let description = [{
    Performs element-wise logarithm plus one operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one
  }];
}

def PPHLO_LogisticOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"logistic", [SameOperandsAndResultType],
                                            PPHLO_FpOrFxpTensor> {
  let summary = "Reciprocal operator";
  let description = [{
    Performs element-wise logistic operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#logistic
  }];
}

def PPHLO_NegOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"negate", [SameOperandsAndResultType,
                                                       DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]
                                            , PPHLO_NoComplexTensor> {
  let summary = "Negation operator";
  let description = [{
    Performs element-wise negation of `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate
  }];
}

def PPHLO_NotOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"not", [SameOperandsAndResultType,
                                                    DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>],
                                                    PPHLO_BoolOrIntTensor> {
  let summary = "Not operator";
  let description = [{
    Performs element-wise NOT of tensor `operand` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not
  }];
}

def PPHLO_PopcntOp : PPHLO_UnaryElementwiseOp<"popcnt", [Pure, SameOperandsAndResultType], PPHLO_IntTensor> {
  let summary = "Popcnt operator, ties away from zero";
  let description = [{
    Performs element-wise count of the number of bits set in the `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt
  }];

  let arguments = (ins
      PPHLO_IntTensor: $operand,
      OptionalAttr<I64Attr>: $bits
  );
}

def PPHLO_RoundOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"round_nearest_afz", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Round operator, ties away from zero";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties away from zero,
    on the `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_afz
  }];
}

def PPHLO_RoundNearestEvenOp: PPHLO_UnaryElementwiseOpWithTypeInfer<"round_nearest_even",
    [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "RoundNearestEven operation";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties
    towards the even integer, on the `operand` tensor and produces a `result`
    tensor.

    Ref:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_even
    ```
  }];
}

def PPHLO_RsqrtOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"rsqrt", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Reciprocal of square-root operator";
  let description = [{
    Performs element-wise reciprocal square root operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rsqrt
  }];
}

def PPHLO_SignOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"sign", [SameOperandsAndResultType,
                                                     DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>],
                               PPHLO_SignedIntOrFpOrFxp> {
  let summary = "Sign operator";
  let description = [{
    Returns the sign of the `operand` element-wise and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    PPHLO Extension: when `ignore_zero` is set to true, sign does not enforce sign(0) to 0
  }];

  let arguments = (ins
      PPHLO_SignedIntOrFpOrFxp: $operand,
      DefaultValuedAttr<BoolAttr, "false">: $ignore_zero
  );
}

def PPHLO_SineOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"sine", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "sine operator";
  let description = [{
    Performs element-wise sine operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sine
  }];
}

def PPHLO_SqrtOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"sqrt", [SameOperandsAndResultType, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>], PPHLO_FpOrFxpTensor> {
  let summary = "Square-root operator";
  let description = [{
    Performs element-wise square root operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sqrt
  }];
}

def PPHLO_TanhOp
    : PPHLO_UnaryElementwiseOpWithTypeInfer<"tanh", [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Tanh operator";
  let description = [{
    Performs element-wise hyperbolic tangent operation on `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tanh
  }];
}

//===----------------------------------------------------------------------===//
// pphlo binary elementwise op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_AddOp
    : PPHLO_BinaryElementwiseOpWithTypeInfer<"add", [Commutative, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>], PPHLO_NoComplexTensor> {
  let summary = "Addition operator";
  let description = [{
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add
  }];

  let hasCanonicalizer = 1;
}

def PPHLO_DivOp : PPHLO_BinaryElementwiseOpWithTypeInfer<"divide", [], PPHLO_NoComplexTensor> {
  let summary = "Division operator";
  let description = [{
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide
  }];
  let hasCanonicalizer = 1;
}

def PPHLO_MaxOp
    : PPHLO_BinaryElementwiseOpWithTypeInfer<"maximum", [Commutative], PPHLO_NoComplexTensor> {
  let summary = "Maximum operator";
  let description = [{
    Performs element-wise max operation on tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum
  }];
  let hasCanonicalizer = 1;
}

def PPHLO_MinOp
    : PPHLO_BinaryElementwiseOpWithTypeInfer<"minimum", [Commutative], PPHLO_NoComplexTensor> {
  let summary = "Minimum operator";
  let description = [{
    Performs element-wise min operation on tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum
    }];
  let hasCanonicalizer = 1;
}

def PPHLO_MulOp
    : PPHLO_BinaryElementwiseOpWithTypeInfer<"multiply", [Commutative,
                                                          DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>],
                                                          PPHLO_NoComplexTensor> {
  let summary = "Multiplication operator";
  let description = [{
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PPHLO_PowOp : PPHLO_BinaryElementwiseOpWithTypeInfer<"power", [], PPHLO_NoComplexTensor> {
  let summary = "Power operator";
  let description = [{
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power
  }];
}

def PPHLO_RemOp : PPHLO_BinaryElementwiseOpWithTypeInfer<"remainder", [], PPHLO_NoComplexTensor> {
  let summary = "Remainder operator";
  let description = [{
    Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder
  }];
}

def PPHLO_Atan2Op : PPHLO_BinaryElementwiseOpWithTypeInfer<"atan2", [], PPHLO_FpOrFxpTensor> {
  let summary = "Atan2 operator";
  let description = [{
    Performs element-wise atan2 operation on `lhs` tensor and `rhs` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#atan2
  }];
}

def PPHLO_ShiftLeftOp
    : PPHLO_BinaryElementwiseOpWithTypeInfer<"shift_left", [DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>], PPHLO_IntTensor> {
  let summary = "Shift Left operator";
  let description = [{
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs` number of bits and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left
  }];
}

def PPHLO_ShiftRightArithmeticOp
    : PPHLO_BinaryElementwiseOpWithTypeInfer<"shift_right_arithmetic", [DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>], PPHLO_IntTensor> {
  let summary = "Shift right arithmetic operator";
  let description = [{
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor by `rhs` number of bits and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic
  }];
}

def PPHLO_ShiftRightLogicalOp
    : PPHLO_BinaryElementwiseOpWithTypeInfer<"shift_right_logical", [DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>], PPHLO_IntTensor> {
  let summary = "Shift right logical operator";
  let description = [{
    Performs element-wise logical right-shift operation on the `lhs` tensor by `rhs` number of bits and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical
  }];
}

def PPHLO_SubtractOp : PPHLO_BinaryElementwiseOpWithTypeInfer<"subtract", [], PPHLO_NoComplexTensor> {
  let summary = "Subtraction operator";
  let description = [{
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract
  }];
}

//===----------------------------------------------------------------------===//
// pphlo binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_AndOp : PPHLO_BinaryLogicalElementwiseOp<"and"> {
  let summary = "And operator";
  let description = [{
    Performs element-wise AND of two tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and
  }];
  let hasFolder = 1;
}

def PPHLO_OrOp : PPHLO_BinaryLogicalElementwiseOp<"or"> {
  let summary = "Or operator";
  let description = [{
    Performs element-wise OR of two tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or
  }];
  let hasFolder = 1;
}

def PPHLO_XorOp : PPHLO_BinaryLogicalElementwiseOp<"xor"> {
  let summary = "Xor operator";
  let description = [{
    Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// pphlo comparison op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_EqualOp: PPHLO_ComparisionOp<"equal", [Commutative, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Equal comparison operator";
  let description = [{
    Performs element-wise equal comparison of `lhs` and `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare
  }];
  let hasCanonicalizer = 1;
}

def PPHLO_NotEqualOp: PPHLO_ComparisionOp<"not_equal", [Commutative]> {
  let summary = "Not-equal comparison operator";
  let description = [{
    Performs element-wise not equal comparison of `lhs` and `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare
  }];
}

def PPHLO_GreaterEqualOp: PPHLO_ComparisionOp<"greater_equal"> {
  let summary = "greater_equal comparison operator";
  let description = [{
    Performs element-wise greater equal comparison of `lhs` and `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare
  }];
}

def PPHLO_GreaterOp: PPHLO_ComparisionOp<"greater"> {
  let summary = "greater comparison operator";
  let description = [{
    Performs element-wise greater comparison of `lhs` and `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare
  }];
}

def PPHLO_LessEqualOp: PPHLO_ComparisionOp<"less_equal"> {
  let summary = "less_equal comparison operator";
  let description = [{
    Performs element-wise less equal comparison of `lhs` and `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare
  }];
}

def PPHLO_LessOp: PPHLO_ComparisionOp<"less", [DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "less comparison operator";
  let description = [{
    Performs element-wise less comparison of `lhs` and `rhs` tensors and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare
  }];
}

//===----------------------------------------------------------------------===//
// pphlo RNG op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_RngOp
    : PPHLO_Op<"rng", [SameOperandsAndResultElementType]> {
  let summary = "RNG with uniform distribution.";
  let description = [{
    Constructs an output of a given shape with random numbers generated
    following the uniform distribution over the interval `[a,b)`. The parameters
    and output element type have to be an integral type or a
    fixed point type, and the types have to be consistent.

    Ref https://www.tensorflow.org/xla/operation_semantics#rnguniform.
  }];
  let arguments = (ins PPHLO_Tensor : $a, PPHLO_Tensor : $b);
  let results = (outs PPHLO_Tensor);
}

//===----------------------------------------------------------------------===//
// pphlo other op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_BitcastConvertOp : PPHLO_UnaryElementwiseOp<"bitcast_convert", [Pure,
                                                                          DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]
                                                                       , PPHLO_Tensor> {
  let summary = "BitcastConvert operator";
  let description =[{
    Performs a bitcast operation on operand tensor and produces a result tensor where the bits of the entire
    `operand` tensor are reinterpreted using the type of the `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def PPHLO_BroadcastOp
    : PPHLO_Op<"broadcast", [Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Broadcast operator";
  let description = [{
    Expands the dimensions and/or rank of an input tensor by duplicating the data in the `operand` tensor and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim
  }];
  let arguments = (ins PPHLO_AnyTensor: $operand,
                       DenseI64ArrayAttr: $broadcast_dimensions);

  let results = (outs PPHLO_AnyTensor);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $operand `,` `dims` `=` $broadcast_dimensions
      attr-dict `:` functional-type(operands, results)
  }];
}

def PPHLO_ClampOp
    : PPHLO_Op<"clamp", [Pure, SameOperandsAndResultShape,
                         DeclareOpInterfaceMethods<InferTypeOpInterface>,
                         DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let summary = "Clamp operator";
  let description = [{
    Clamps every element of the `operand` tensor between a `minimum` and `maximum` value and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp
  }];

  let arguments = (ins PPHLO_NoComplexTensor: $min,
                       PPHLO_NoComplexTensor: $operand,
                       PPHLO_NoComplexTensor: $max
                  );
  let results = (outs PPHLO_NoComplexTensor: $result);

  let assemblyFormat = [{
    $min `,` $operand `,` $max attr-dict
      `:` custom<SameOperandsAndResultType>(type($min), type($operand), type($max), type($result))
  }];
}

def PPHLO_ConcatenateOp
    : PPHLO_WithShapeInferOp<"concatenate",
               [Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Concatenate op";
  let description = [{
    Concatenates `inputs` along `dimension` dimension in the same order as the given arguments and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate
  }];

  let arguments = (ins Variadic<PPHLO_AnyTensor>: $inputs,
                       I64Attr: $dimension);

  let results = (outs PPHLO_AnyTensor);
  let hasVerifier = 1;
  let assemblyFormat = [{
     operands `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
  }];
}

def PPHLO_ConvolutionOp : PPHLO_Op<"convolution", [Pure]> {
  let summary = "Convolution operator";
  let description = [{
    Computes dot products between windows of `lhs` and slices of `rhs` and produces `result`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution
  }];
  let arguments = !con(
    (ins
       PPHLO_Tensor:$lhs,
       PPHLO_Tensor:$rhs),
    ConvolutionAttributes.attributes);

  let results = (outs PPHLO_Tensor);

  let assemblyFormat = [{
    `(`operands`)`
       `dim_numbers` `=` custom<ConvolutionDimensions>($dimension_numbers) `,`
       `window` `=` `{` custom<WindowAttributes>($window_strides) `}`
       attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

def PPHLO_DotOp : PPHLO_Op<"dot", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                         DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let summary = "Dot operator";
  let description = [{
    Performs dot products between vectors, vector/matrix and matrix/matrix
    multiplication.

    Ref https://www.tensorflow.org/xla/operation_semantics#dot.
  }];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_Tensor);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

def PPHLO_DotGeneralOp: PPHLO_Op<"dot_general", [Pure]> {
  let summary = "General Dot operator";
  let description = [{
    Computes dot products between slices of `lhs` and slices of `rhs` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
  }];
  let arguments = (ins
    PPHLO_Tensor:$lhs,
    PPHLO_Tensor:$rhs,
    DotDimensionNumbers:$dot_dimension_numbers
  );

  let results = (outs PPHLO_Tensor);
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` custom<DotDimensionNumbers>($dot_dimension_numbers) attr-dict
      `:` functional-type(operands, results)
  }];
}

def PPHLO_DynamicSliceOp: PPHLO_WithShapeInferOp<"dynamic_slice", [Pure, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Dynamic Slice operator";
  let description = [{
    Extracts a slice from the `operand` using dynamically-computed starting indices and produces a `result` tensor.

    `start_indices` contain the starting indices of the slice for each dimension subject to potential adjustment,
    and `slice_sizes` contain the sizes of the slice for each dimension.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice
  }];
  let arguments = (ins PPHLO_AnyTensor:$operand,
                       Variadic<PPHLO_ScalarIntTensor>:$start_indices,
                       DenseI64ArrayAttr:$slice_sizes
  );

  let results = (outs PPHLO_AnyTensor:$result);

  let assemblyFormat = [{
    operands `sizes` `=` $slice_sizes
      attr-dict `:` functional-type(operands, results)
  }];
}

def PPHLO_DynamicUpdateSliceOp: PPHLO_WithShapeInferOp<"dynamic_update_slice",
      [Pure, AllShapesMatch<["operand", "result"]>, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Dynamic Update Slice operator";
  let description = [{
    Produces a `resul`t tensor which is equal to the `operand` tensor except that the slice starting at `start_indices`
    is updated with the values in `update`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_update_slice
  }];
  let arguments = (ins PPHLO_AnyTensor:$operand,
                       PPHLO_AnyTensor:$update,
                       Variadic<PPHLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs PPHLO_AnyTensor:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def HLO_PadOp
    : PPHLO_WithShapeInferOp<"pad",
    [Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<InferShapedTypeOpInterface>, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Pad operator";
  let description = [{
    Expands `operand` by padding around the tensor as well as between the elements of the tensor with the given `padding_value`.

    `edge_padding_low` and `edge_padding_high` specify the amount of padding added at the low-end (next to index 0) and the high-end (next to the highest index)
    of each dimension respectively. The amount of padding can be negative, where the absolute value of negative padding indicates the number of elements to remove from the specified dimension.

    `interior_padding` specifies the amount of padding added between any two elements in each dimension which may not be negative.
    Interior padding occurs before edge padding such that negative edge padding will remove elements from the interior-padded operand.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad
  }];

  let arguments = (ins PPHLO_AnyTensor: $operand,
                       PPHLO_AnyTensor: $padding_value,
                       DenseI64ArrayAttr: $edge_padding_low,
                       DenseI64ArrayAttr: $edge_padding_high,
                       DenseI64ArrayAttr: $interior_padding);

  let results = (outs PPHLO_AnyTensor);
  let assemblyFormat = [{
    $operand `,` $padding_value `,`
      `low` `=` $edge_padding_low `,`
      `high` `=` $edge_padding_high `,`
      `interior` `=` $interior_padding
      attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

def PPHLO_ReduceOp : PPHLO_Op<"reduce", [
  RecursiveMemoryEffects, SameVariadicOperandSize,
  SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "Reduce operator";
  let description = [{
    Applies a reduction function `body` to `inputs` and `init_values` along the `dimensions` and produces `results` tensors.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce
  }];
  let arguments = (ins
    Variadic<PPHLO_Tensor>:$inputs,
    Variadic<PPHLO_Tensor>:$init_values,
    DenseI64ArrayAttr:$dimensions
  );

  let results = (outs Variadic<PPHLO_Tensor>);

  let regions = (region SizedRegion<1> : $body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PPHLO_ReshapeOp
    : PPHLO_Op<"reshape", [Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Reshape operator";
  let description = [{
    Performs reshape of `operand` tensor to a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape
  }];
  let arguments = (ins PPHLO_AnyTensor: $operand);
  let results = (outs PPHLO_AnyTensor: $result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def PPHLO_ReduceWindowOp : PPHLO_Op<"reduce_window", [
  RecursiveMemoryEffects, SameVariadicOperandSize,
  SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "ReduceWindow operator";
  let description = [{
    Applies a reduction function `body` to windows of `inputs` and `init_values` and produces `results`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window
  }];

  let arguments = (ins
    Variadic<PPHLO_Tensor>:$inputs,
    Variadic<PPHLO_Tensor>:$init_values,
    DenseI64ArrayAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the input dimensions.
    OptionalAttr<DenseI64ArrayAttr>:$window_strides,
    OptionalAttr<DenseI64ArrayAttr>:$window_dilations
  );

  let results = (outs Variadic<PPHLO_Tensor>);
  let regions = (region SizedRegion<1> : $body);
}

def PPHLO_ReverseOp
    : PPHLO_Op<"reverse", [Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Reverse operator";
  let description = [{
    Reverses the order of elements in the `operand` along the specified `dimensions` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reverse
  }];

  let arguments = (ins PPHLO_AnyTensor: $operand,
                       DenseI64ArrayAttr: $dimensions
  );

  let results = (outs PPHLO_AnyTensor: $result);

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` `dims` `=` $dimensions attr-dict `:` type($result)
  }];
}

def PPHLO_SelectOp: PPHLO_Op<"select", [Pure]> {
  let summary = "Select operator";
  let description = [{
    Produces a `result` tensor where each element is selected from `on_true` or `on_false` tensor based on the value of the corresponding element of `pred`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#Select
  }];
  let arguments = (ins
                  PPHLO_BoolTensor: $pred,
                  PPHLO_Tensor: $on_true,
                  PPHLO_Tensor: $on_false
  );

  let results = (outs PPHLO_Tensor: $result);
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def PPHLO_SliceOp : PPHLO_WithShapeInferOp<"slice", [
  Pure, SameOperandsAndResultElementType,
  AllMatchSameOperatorTrait<["start_indices", "limit_indices",
      "strides"], "$_self.size()", "size"> /*slice_c2*/, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>
]> {
  let description = [{
    Extracts a slice from the `operand` using statically-computed starting indices and produces a `result` tensor.
    `start_indices` contain the starting indices of the slice for each dimension,
    `limit_indices` contain the ending indices (exclusive) for the slice for each dimension,
    and `strides` contain the strides for each dimension.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice
  }];
  let arguments = (ins PPHLO_AnyTensor: $operand,
                       DenseI64ArrayAttr: $start_indices,
                       DenseI64ArrayAttr: $limit_indices,
                       DenseI64ArrayAttr: $strides);

  let results = (outs PPHLO_AnyTensor);
  let hasVerifier = 1;
  let hasFolder = 1;

  let assemblyFormat = [{
    $operand custom<SliceRanges>($start_indices, $limit_indices, $strides)
      attr-dict `:` functional-type(operands, results)
  }];
}

def PPHLO_SortOp
    : PPHLO_Op<"sort", [RecursiveMemoryEffects, SameOperandsAndResultShape]> {
  let summary = "Sort operator";
  let description = [{
    Sorts 1-dimensional slices of inputs along the dimension `dimension` together, according to a `comparator` and produces `results`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sort
  }];

  let arguments = (ins
    Variadic<PPHLO_Tensor>:$operands,
    DefaultValuedAttr<I64Attr, "-1">:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$is_stable
  );

  let regions = (region SizedRegion<1>:$comparator);
  let results = (outs Variadic<PPHLO_Tensor>);
}


def PPHLO_TransposeOp
    : PPHLO_WithShapeInferOp<"transpose", [Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Transpose operator";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose
  }];
  let arguments = (ins PPHLO_AnyTensor : $operand, DenseI64ArrayAttr : $permutation);
  let results = (outs PPHLO_AnyTensor);
  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` `dims` `=` $permutation
      attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// pphlo control flow op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_CaseOp: PPHLO_Op<"case", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]> {
  let summary = "Switch-Case operator";
  let description = [{
    Produces the output from executing exactly one function from `branches` depending on the value of `index`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case
  }];

  let arguments = (ins PPHLO_IntTensor:$index);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let results = (outs Variadic<PPHLO_Tensor>);
}

def PPHLO_IfOp : PPHLO_Op<"if", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "If operator";

  let description = [{
    Produces the output from executing exactly one function from `true_branch` or `false_branch` depending on the value of `pred`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if
  }];

  let arguments = (ins PPHLO_BoolTensor:$condition);

  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);

  let results = (outs Variadic<PPHLO_Tensor>);
}

def PPHLO_WhileOp : PPHLO_Op<"while", [
  RecursiveMemoryEffects, PPHLO_PairwiseSameOperandAndResultType, SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "While operator";
  let description = [{
    Produces the output from executing `body` function 0 or more times while the `cond` function outputs true.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while
  }];
  let arguments = (ins Variadic<PPHLO_Tensor> : $args);
  let regions = (region SizedRegion<1> : $cond, SizedRegion<1> : $body);
  let results = (outs Variadic<PPHLO_Tensor>);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// pphlo complex op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_ComplexOp: PPHLO_BinaryElementwiseOp<"complex", [Pure,
    SameOperandsElementType /*complex_c1*/,
    SameOperandsAndResultShape /*complex_c2*/], PPHLO_FpOrFxpTensor, PPHLO_AllComplexTensor> {
  let summary = "Complex operator";
  let description = [{
    Performs element-wise conversion to a complex value from a pair of real and
    imaginary values, `lhs` and `rhs`, and produces a `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#complex
  }];
}

def PPHLO_RealOp: PPHLO_UnaryElementwiseOp<"real",
    [Pure], PPHLO_FpOrFxpOrComplexTensor, PPHLO_FpOrFxpTensor> {
  let summary = "Real operator";
  let description = [{
    Extracts the real part, element-wise, from the `operand` and produces a
    `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#real
  }];
}

def PPHLO_ImagOp: PPHLO_UnaryElementwiseOp<"imag",
    [Pure], PPHLO_FpOrFxpOrComplexTensor, PPHLO_FpOrFxpTensor> {
  let summary = "Imag operator";
  let description = [{
    Extracts the imaginary part, element-wise, from the `operand` and produces a
    `result` tensor.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#imag
  }];
}

//===----------------------------------------------------------------------===//
// pphlo custom call op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_CustomCallOp: PPHLO_Op<"custom_call",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "CustomCall operator";
  let description = [{
    Encapsulates an implementation-defined operation `call_target_name` that
    takes `inputs` and `called_computations` and produces `results`.

    Ref https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call
  }];

  let arguments = (ins
    Variadic<PPHLO_Tensor>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$allow_float
  );
  let results = (outs Variadic<PPHLO_Tensor>);
  let assemblyFormat = [{
    custom<CustomCallTarget>($call_target_name) `(` $inputs `)`
      attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// pphlo extended op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_EpsilonOp: PPHLO_Op<"epsilon", [Pure]> {
  let summary = "Epsilon operator";
  let description = [{
    Produces a tensor that representations runtime epsilon.
  }];
  let results = (outs PPHLO_FpOrFxpTensor: $output);
  let assemblyFormat = [{
      attr-dict `:` type($output)
  }];
}

def PPHLO_ReciprocalOp: PPHLO_UnaryElementwiseOpWithTypeInfer<"reciprocal",
                          [SameOperandsAndResultType], PPHLO_FpOrFxpTensor> {
  let summary = "Reciprocal operator";
  let description = [{
    Performs element-wise reciprocal of `operand` tensor and produces a `result` tensor.
  }];
}

def PPHLO_ReturnOp : PPHLO_Op<"return", [Pure, Terminator]> {
  let summary = "Return operator";
  let description = [{
    Terminates a region and returns `results`.
  }];

  let arguments = (ins Variadic<PPHLO_Tensor> : $results);

  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

def PPHLO_SimpleSortOp
    : PPHLO_Op<"simple_sort", [RecursiveMemoryEffects, SameOperandsAndResultShape]> {
  let summary = "Sort operator";
  let description = [{
    Sorts the given `operands` at the given `dimension` with the given
    `mode`.
  }];
  let arguments = (ins
    Variadic<PPHLO_Tensor>:$operands,
    DefaultValuedAttr<I64Attr, "-1">:$dimension,
    DefaultValuedAttr<I64Attr, "1">:$num_keys,
    PPHLO_SortDirectionAttr: $sort_direction
  );

  let results = (outs Variadic<PPHLO_Tensor>);

  let assemblyFormat = [{
    operands $sort_direction `,` `dim` `=` $dimension `,` `num_keys` `=` $num_keys
      attr-dict `:` functional-type(operands, results)
  }];
}

def PPHLO_TruncOp : PPHLO_Op<"truncate", [Pure, Elementwise, SameOperandsAndResultShape,
                                          DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>]> {
  let summary = "Truncation operator";
  let description = [{
    Performs element-wise truncation operation on `operand` tensor and produces a
    `result` tensor.
  }];
  let arguments = (ins
      PPHLO_FxpTensor : $operand,
      DefaultValuedAttr<PPHLO_SignTypeAttr, "SignType::Unknown">: $sign
  );
  let results = (outs PPHLO_FxpTensor);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PPHLO_PrefixOrOp: PPHLO_UnaryElementwiseOp<"prefix_or",
                        [Pure, SameOperandsAndResultType,
                         DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>],
                        PPHLO_IntTensor> {
  let summary = "Prefix or operator";
  let description = [{
    Performs element-wise prefix or of `operand` tensors and produces a `result` tensor.
  }];
}

def PPHLO_BitRevOp: PPHLO_UnaryElementwiseOp<"bitrev",
                        [Pure, SameOperandsAndResultType,
                         DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>],
                        PPHLO_IntTensor> {
  let summary = "Bitreverse operator";
  let description = [{
    Performs element-wise bitreverse or of `operand` tensors and produces a `result` tensor.
  }];
  let arguments = (ins
    PPHLO_IntTensor:$operand,
    I64Attr: $start,
    I64Attr: $end
  );
}

def PPHLO_BitDeintlOp: PPHLO_UnaryElementwiseOp<"bitdeintl",
                        [Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<InferFxpRangeInterface, ["inferResultRanges"]>],
                        PPHLO_IntTensor> {
  let summary = "Bit-deinterleave operator";
  let description = [{
    Performs element-wise bit-deinterleave of `operand` tensors and produces a `result` tensor.
  }];
}

def PPHLO_BitParityOp: PPHLO_UnaryElementwiseOp<"bitparity",
                        [Pure], PPHLO_IntTensor, PPHLO_BoolTensor> {
  let summary = "Bitparity operator";
  let description = [{
    Performs element-wise bitparity or of `operand` tensors and produces a `result` tensor.
  }];
  let arguments = (ins
    PPHLO_IntTensor:$operand,
    I64Attr: $bits
  );
}

#endif  // SPU_DIALECT_PPHLO_OPS
