//
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
include "pphlo_types.td"
include "pphlo_attrs.td"

//===----------------------------------------------------------------------===//
// Common traits
//===----------------------------------------------------------------------===//

class PPHLO_NativeOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::pphlo::OpTrait";
}

def PPHLO_PairwiseSameOperandAndResultType
    : PPHLO_NativeOpTrait<"PairwiseSameOperandAndResultType">;

class PPHLO_Op<string mnemonic, list<Trait> traits>
    : Op<PPHlo_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// pphlo nullary op definitions.
//===----------------------------------------------------------------------===//

def PPHLO_ConstOp : PPHLO_Op<"constant", [ConstantLike, NoSideEffect]> {
  let summary = "Constant operator";
  let description = [{Create a constant value.}];
  let arguments = (ins ElementsAttr : $value);
  let results = (outs PPHLO_Tensor : $output);
  let builders = [OpBuilder<(ins "Attribute" : $value)>];
  let skipDefaultBuilders = 1;
  let hasFolder = 1;
}

def PPHLO_IotaOp : PPHLO_Op<"iota", [NoSideEffect]> {
  let summary = "Iota operator";
  let description = [{
    Creates a rank 1 array of values starting at zero and incrementing by one.

    See https://www.tensorflow.org/xla/operation_semantics#iota
  }];
  let arguments = (ins I64Attr : $iota_dimension);

  let results = (outs PPHLO_IntTensor : $output);
}

//===----------------------------------------------------------------------===//
// pphlo unary elementwise op definitions.
//===----------------------------------------------------------------------===//
class PPHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
                               Type TensorType>
    : PPHLO_Op<mnemonic, traits #[Elementwise, SameOperandsAndResultShape]> {
  let arguments = (ins TensorType : $operand);
  let results = (outs TensorType);
}

def PPHLO_ConvertOp
    : PPHLO_UnaryElementwiseOp<"convert", [NoSideEffect], PPHLO_Tensor> {
  let summary = "Convert operator";
  let description = [{
    Performs element-wise conversion of values from one type to another,
    e.g.fxp to int.

    See
    https://www.tensorflow.org/xla/operation_semantics#convertelementtype.
  }];
  let builders = [OpBuilder<(ins "Value"
                             : $operand, "Type"
                             : $result_element_ty)>];

  let hasFolder = 1;
}

def PPHLO_SqrtOp
    : PPHLO_UnaryElementwiseOp<"sqrt", [NoSideEffect], PPHLO_FpTensor> {
  let summary = "Square-root operator";
  let description = [{
    Returns `sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_TanhOp
    : PPHLO_UnaryElementwiseOp<"tanh", [NoSideEffect], PPHLO_FpTensor> {
  let summary = "Tanh operator";
  let description = [{
    Returns `tanh(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_NegOp
    : PPHLO_UnaryElementwiseOp<"negate", [NoSideEffect], PPHLO_Tensor> {
  let summary = "Negation operator";
  let description = [{
    Returns `- operand` element -
    wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_ExpOp
    : PPHLO_UnaryElementwiseOp<"exponential", [NoSideEffect], PPHLO_FpTensor> {
  let summary = "Exponential operator";
  let description = [{
    Returns `e ^ (operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_LogOp
    : PPHLO_UnaryElementwiseOp<"log", [NoSideEffect], PPHLO_FpTensor> {
  let summary = "Log operator";
  let description = [{
    Returns `log(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_Log1pOp : PPHLO_UnaryElementwiseOp<"log_plus_one", [NoSideEffect],
                                             PPHLO_FpTensor> {
  let summary = "Log1p operator";
  let description = [{
    Returns `log(operand + 1)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_CeilOp
    : PPHLO_UnaryElementwiseOp<"ceil", [NoSideEffect], PPHLO_FpTensor> {
  let summary = "Ceil operator";
  let description = [{
    Returns `Ceil(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_FloorOp
    : PPHLO_UnaryElementwiseOp<"floor", [NoSideEffect], PPHLO_FpTensor> {
  let summary = "Floor operator";
  let description = [{
    Returns `Floor(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_AbsOp
    : PPHLO_UnaryElementwiseOp<"abs", [NoSideEffect], PPHLO_Tensor> {
  let summary = "Absolute value operator";
  let description = [{
    Returns `abs(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_ReciprocalOp
    : PPHLO_UnaryElementwiseOp<"reciprocal",
                               [NoSideEffect, SameOperandsAndResultType],
                               PPHLO_FpTensor> {
  let summary = "Reciprocal operator";
  let description = [{
    Returns `1.0 / operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
  let hasFolder = 1;
}

def PPHLO_LogisticOp
    : PPHLO_UnaryElementwiseOp<"logistic",
                               [NoSideEffect, SameOperandsAndResultType],
                               PPHLO_FpTensor> {
  let summary = "Reciprocal operator";
  let description = [{
    Returns `logistic(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def PPHLO_NotOp
    : PPHLO_UnaryElementwiseOp<"not", [NoSideEffect, SameOperandsAndResultType],
                               PPHLO_IntTensor> {
  let summary = "Not operator";
  let description = [{
    Returns `!operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

//===----------------------------------------------------------------------===//
// pphlo binary elementwise op definitions.
//===----------------------------------------------------------------------===//

class PPHLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits>
    : PPHLO_Op<mnemonic, traits #[SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);

  let results = (outs PPHLO_Tensor);
}

def PPHLO_SubOp : PPHLO_BinaryElementwiseOp<"subtract", [NoSideEffect]> {
  let summary = "Subtraction operator";
  let description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_MaxOp
    : PPHLO_BinaryElementwiseOp<"maximum", [Commutative, NoSideEffect]> {
  let summary = "Maximum operator";
  let description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_MinOp
    : PPHLO_BinaryElementwiseOp<"minimum", [Commutative, NoSideEffect]> {
  let summary = "Minimum operator";
  let description = [{
    Returns `min(lhs, rhs)` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
    }];
}

def PPHLO_DivOp : PPHLO_BinaryElementwiseOp<"divide", [NoSideEffect]> {
  let summary = "Division operator";
  let description = [{
    Returns `lhs / rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_AddOp
    : PPHLO_BinaryElementwiseOp<"add", [Commutative, NoSideEffect]> {
  let summary = "Addition operator";
  let description = [{
    Returns `lhs + rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_MulOp
    : PPHLO_BinaryElementwiseOp<"multiply", [Commutative, NoSideEffect]> {
  let summary = "Multiplication operator";
  let description = [{
    Returns `lhs * rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_PowOp : PPHLO_BinaryElementwiseOp<"power", [NoSideEffect]> {
  let summary = "Power operator";
  let description = [{
    Returns `lhs ^ rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_RemOp : PPHLO_BinaryElementwiseOp<"remainder", [NoSideEffect]> {
  let summary = "Remainder operator";
  let description = [{
    Returns `lhs % rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_ShiftLeftOp
    : PPHLO_BinaryElementwiseOp<"shift_left",
                                [NoSideEffect]> {
  let summary = "Shift Left operator";
  let description = [{
    Returns `lhs << rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_ShiftRightLogicalOp
    : PPHLO_BinaryElementwiseOp<"shift_right_logical",
                                [NoSideEffect]> {
  let summary = "Shift right logical operator";
  let description = [{
    Returns logical `lhs >> rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_ShiftRightArithmeticOp
    : PPHLO_BinaryElementwiseOp<"shift_right_arithmetic",
                                [NoSideEffect]> {
  let summary = "Shift right arithmetic operator";
  let description = [{
    Returns arithmetic `lhs >> rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// pphlo binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//
class PPHLO_BinaryLogicalElementwiseOp<string mnemonic>
    : PPHLO_BinaryElementwiseOp<
          mnemonic, [Commutative, NoSideEffect, SameOperandsAndResultShape]> {
  let arguments = (ins PPHLO_IntTensor : $lhs, PPHLO_IntTensor : $rhs);
}

def PPHLO_AndOp : PPHLO_BinaryLogicalElementwiseOp<"and"> {
  let summary = "Logical and";
  let description = [{
    Returns `logical_and(lhs, rhs)` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_OrOp : PPHLO_BinaryLogicalElementwiseOp<"or"> {
  let summary = "Logical or";
  let description = [{
    Returns `logical_or(lhs, rhs)` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def PPHLO_XorOp : PPHLO_BinaryLogicalElementwiseOp<"xor"> {
  let summary = "Logical xor";
  let description = [{
    Returns `logical_xor(lhs, rhs)` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// pphlo comparison op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_EqualOp
    : PPHLO_Op<"equal",
               [NoSideEffect, SameOperandsAndResultShape, Elementwise]> {
  let summary = "Equal comparison operator";
  let description = [{
    Returns `lhs` == `rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_IntTensor);

  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

def PPHLO_NotEqualOp
    : PPHLO_Op<"not_equal",
               [NoSideEffect, SameOperandsAndResultShape, Elementwise]> {
  let summary = "Not-equal comparison operator";
  let description = [{
    Returns `lhs` != `rhs` element-wise.
  
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_IntTensor);

  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

def PPHLO_GreaterEqualOp
    : PPHLO_Op<"greater_equal",
               [NoSideEffect, SameOperandsAndResultShape, Elementwise]> {
  let summary = "greater_equal comparison operator";
  let description = [{
    Returns `lhs` >= `rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_IntTensor);

  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

def PPHLO_GreaterOp
    : PPHLO_Op<"greater",
               [NoSideEffect, SameOperandsAndResultShape, Elementwise]> {
  let summary = "greater comparison operator";
  let description = [{elementwise `lhs` > `rhs`.}];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_IntTensor);

  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

def PPHLO_LessEqualOp
    : PPHLO_Op<"less_equal",
               [NoSideEffect, SameOperandsAndResultShape, Elementwise]> {
  let summary = "less_equal comparison operator";
  let description = [{elementwise `lhs` <= `rhs`.}];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_IntTensor);

  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

def PPHLO_LessOp
    : PPHLO_Op<"less",
               [NoSideEffect, SameOperandsAndResultShape, Elementwise]> {
  let summary = "less comparison operator";
  let description = [{
    Returns `lhs` < `rhs` element-wise.
    
    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_IntTensor);

  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

//===----------------------------------------------------------------------===//
// pphlo RNG op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_RngUniformOp
    : PPHLO_Op<"rng_uniform", [SameOperandsAndResultElementType]> {
  let summary = "RNG with uniform distribution.";
  let description = [{
    Constructs an output of a given shape with random numbers generated
    following the uniform distribution over the interval `[a,b)`. The parameters
    and output element type have to be an integral type or a
    fixed point type, and the types have to be consistent.

    See https://www.tensorflow.org/xla/operation_semantics#rnguniform.
  }];
  let arguments = (ins PPHLO_Tensor : $a, PPHLO_Tensor : $b);

  let results = (outs PPHLO_Tensor);
}

//===----------------------------------------------------------------------===//
// pphlo other op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_BroadcastOp
    : PPHLO_Op<"broadcast", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Broadcast a tensor into the given shape by adding dimensions.";
  let description = [{
    Broadcasts the `operand` tensor to a higher rank. This is not the limited
    form of broadcasting exposed as the XLA client broadcast op, but rather the
    more powerful "InDim" broadcasting, which is closer to the HLO broadcast op
    and exposed in the XLA client BroadcastInDim method.

    `broadcast_dimensions` maps the operand dimension number to the target shape
    dimension number. It must have the same size as the rank of the operand. The
    mapped dimensions must either be the same size or the dimension being
    broadcast from must be size 1 (degenerate broadcasting).

    For a scalar (0D tensor) operand, `broadcast_dimensions` must be empty. The
    The scalar value will be broadcast to every element in the target shape.

    See https://www.tensorflow.org/xla/broadcasting.
  }];
  let arguments = (ins PPHLO_Tensor
                   : $operand, I64ElementsAttr
                   : $broadcast_dimensions);

  let results = (outs PPHLO_Tensor);
}

def PPHLO_ReshapeOp
    : PPHLO_Op<"reshape", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Reshape operator";
  let description = [{
    Reshapes the dimensions of `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#reshape.
  }];
  let arguments = (ins PPHLO_Tensor : $operand);

  let results = (outs PPHLO_Tensor);
  let hasFolder = 1;
}

def PPHLO_ConcatenateOp
    : PPHLO_Op<"concatenate",
               [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Concatenate op";
  let description = [{
    Concatenates a set of tensors along the specified dimension.

    See https://www.tensorflow.org/xla/operation_semantics#concatenate.
  }];

  let arguments = (ins Variadic<PPHLO_Tensor> : $val, I64Attr : $dimension);

  let results = (outs PPHLO_Tensor);
}

def PPHLO_DotOp : PPHLO_Op<"dot", [NoSideEffect]> {
  let summary = "Dot operator";
  let description = [{
    Performs dot products between vectors, vector/matrix and matrix/matrix
    multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dot.
  }];
  let arguments = (ins PPHLO_Tensor : $lhs, PPHLO_Tensor : $rhs);
  let results = (outs PPHLO_Tensor);
}

def PPHLO_SelectOp
    : PPHLO_Op<"select", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Select operator";
  let description = [{
    Constructs an output tensor from the elements of `on_true` and `on_false`
    based on the values of `pred`. All three operands must be of the same shape
    with the exception of `pred`, which may also be a scalar in which case it is
    broadcasted.

    See https://www.tensorflow.org/xla/operation_semantics#select.
  }];
  let arguments = (ins PPHLO_IntTensor
                   : $pred, PPHLO_Tensor
                   : $on_true, PPHLO_Tensor
                   : $on_false);

  let results = (outs PPHLO_Tensor);
}

def PPHLO_SelectAndScatterOp: PPHLO_Op<"select_and_scatter",
      [RecursiveSideEffects]> {
  let summary = "SelectAndScatter operator";
  let description = [{
    Runs a windowed selection `select` function over `operand` with shape
    `window_dimensions` and stride `window_strides`. This will produce an amount
    of selected locations whose shape matches `source`. These are then scattered
    to the output which is initialized with `init_value`.
    Multiple scattered elements which land in the same output location are
    combined using the `scatter` function.

    See https://www.tensorflow.org/xla/operation_semantics#selectandscatter.
  }];
  let arguments = (ins
    PPHLO_Tensor:$operand,
    PPHLO_Tensor:$source,
    PPHLO_Tensor:$init_value,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);

  let results = (outs PPHLO_Tensor);
}

def PPHLO_ReduceOp : PPHLO_Op<"reduce", [
  RecursiveSideEffects, SameVariadicOperandSize,
  SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "Reduce operator";
  let description = [{
    Returns the result of executing a reduction function on one or more arrays
    in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reduce.
  }];
  let arguments = (ins
    Variadic<PPHLO_Tensor>:$inputs,
    Variadic<PPHLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );

  let results = (outs Variadic<PPHLO_Tensor>);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$init_values,
      "DenseIntElementsAttr":$dimensions)>];

  let regions = (region SizedRegion<1> : $body);
}

def PPHLO_ReduceWindowOp : PPHLO_Op<"reduce_window", [
  RecursiveSideEffects, SameVariadicOperandSize,
  SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "ReduceWindow operator";
  let description = [{
    Returns the result of executing a reduction function over all elements in
    each window of one or more arrays.

    See https://www.tensorflow.org/xla/operation_semantics#reducewindow.
  }];

  let arguments = (ins
    PPHLO_Tensor:$inputs,
    PPHLO_Tensor:$init_values,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the input dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs PPHLO_Tensor);

  let regions = (region SizedRegion<1> : $body);
}

def PPHLO_ReturnOp : PPHLO_Op<"return", [NoSideEffect, Terminator]> {
  let summary = [{
    The `pphlo.return` operation terminates a region and returns values.
  }];

  let arguments = (ins Variadic<PPHLO_Tensor> : $results);
}

def PPHLO_TransposeOp
    : PPHLO_Op<"transpose", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Transpose operator";
  let description = [{
    Permutes the dimensions of `operand` according to the given `permutation`.

    `res_dimensions[i] = operand_dimensions[permutation[i]]`

    See https://www.tensorflow.org/xla/operation_semantics#transpose.
  }];
  let arguments = (ins PPHLO_Tensor : $operand, I64ElementsAttr : $permutation);
  let results = (outs PPHLO_Tensor);
  let hasFolder = 1;
}

def PPHLO_SliceOp : PPHLO_Op<"slice", [
  NoSideEffect, SameOperandsAndResultElementType,
  AllTypesMatch<["start_indices", "limit_indices", "strides"]>
]> {
  let description = [{
    The dynamic shape version of SliceOp. Extracts a sub-array from the input
    array according to start_indices, limit_indices and strides. Expect
    start_indices/limit_indices/strides to be statically shaped and matching
    the rank of the input.

    See https://www.tensorflow.org/xla/operation_semantics#slice
  }];
  let arguments = (ins PPHLO_Tensor
                   : $operand, I64ElementsAttr
                   : $start_indices, I64ElementsAttr
                   : $limit_indices, I64ElementsAttr
                   : $strides);

  let results = (outs PPHLO_Tensor);
}

def PPHLO_DbgPrintOp : PPHLO_Op<"dbg_print", []> {
  let arguments = (ins PPHLO_Tensor : $operand);
}

def PPHLO_ClampOp
    : PPHLO_Op<"clamp", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Clamp operator";
  let description = [{
    Clamps an operand to within the range between a minimum and maximum value.

    Note: All three arrays must be the same shape. Alternatively, as a
          restricted form of broadcasting, min and/or max can be a scalar (0D
          tensor) of the element type of the tensor operand.

    See https://www.tensorflow.org/xla/operation_semantics#clamp.
  }];

  let arguments = (ins PPHLO_Tensor
                   : $min, PPHLO_Tensor
                   : $operand, PPHLO_Tensor
                   : $max);
  let results = (outs PPHLO_Tensor);
}

def PPHLO_BitcastConvertOp : PPHLO_Op<"bitcast_convert", [NoSideEffect]> {
  let summary = "BitcastConvert operator";
  let description =[{
   Similar to a 'tf.bitcast' in TensorFlow, performs an element-wise bitcast
    operation from a data shape to a target shape. The dimensions must match,
    and the conversion is an element-wise one. Bitcast is implemented as a
    low-level cast, so machines with different floating-point representations
    will give different results.

    See https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype.
  }];

  let arguments = (ins PPHLO_Tensor : $operand, I64Attr : $elsize);
  let results = (outs PPHLO_Tensor);
}

def PPHLO_SortOp
    : PPHLO_Op<"sort", [RecursiveSideEffects, SameOperandsAndResultShape]> {
  let summary = "Sort operator";
  let description = [{
    Sorts the given `operands` at the given `dimension` with the given
    `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];
  let arguments = (ins
    Variadic<PPHLO_Tensor>:$operands,
    DefaultValuedAttr<I64Attr, "-1">:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$is_stable
  );

  let regions = (region SizedRegion<1>:$comparator);

  let results = (outs Variadic<PPHLO_Tensor>);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands, CArg<"int64_t", "-1">:$dimension,
      CArg<"bool", "false">:$is_stable)>];
}

def PPHLO_ReverseOp
    : PPHLO_Op<"reverse", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Reverse operator";
  let description = [{
    Reverses the specified dimensions of `operand` according to the given
    `dimensions`.

    See https://www.tensorflow.org/xla/operation_semantics#rev_reverse.
  }];
  let arguments = (ins PPHLO_Tensor : $operand, I64ElementsAttr : $dimensions);

  let results = (outs PPHLO_Tensor);
}

def HLO_PadOp
    : PPHLO_Op<"pad", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Pad operator";
  let description = [{
    Pads the edges of `operand` with the `padding_value` and according to
    the passed configuration.

    See https://www.tensorflow.org/xla/operation_semantics#pad.
  }];
  let arguments = (ins PPHLO_Tensor
                   : $operand, PPHLO_Tensor
                   : $padding_value, I64ElementsAttr
                   : $edge_padding_low, I64ElementsAttr
                   : $edge_padding_high, I64ElementsAttr
                   : $interior_padding);

  let results = (outs PPHLO_Tensor);
}

def PPHLO_GatherOp : PPHLO_Op<"gather", [NoSideEffect]> {
  let summary = "Gather operator";
  let description = [{
    Stitches together several slices of `operand` from offsets specified in
    `start_indices` (each slice at a potentially different runtime offset).

    See https://www.tensorflow.org/xla/operation_semantics#gather.
  }];

  let arguments = (ins
    PPHLO_Tensor:$operand,
    PPHLO_IntTensor:$start_indices,
    GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedAttr<BoolAttr, "false">:$indices_are_sorted
  );

  let results = (outs PPHLO_Tensor);
}

def ConvolutionAttributes {
  dag attributes = (ins
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: zero for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count
  );
}

def PPHLO_ConvOp : PPHLO_Op<"convolution", [NoSideEffect]> {
  let summary = "Convolution operator";
  let description = [{
    Computes a convolution of the kind used in neural networks.

    See https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
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
       `window` `=` `{` custom<WindowAttributes>($window_strides, $padding,
                                                 $lhs_dilation, $rhs_dilation) `}`
       attr-dict `:` functional-type(operands, results)
  }];
}

def PPHLO_DynamicSliceOp: PPHLO_Op<"dynamic-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>]> {
  let summary = "Dynamic Slice operator";
  let description = [{
    Extracts a sub-array from the input array at dynamic start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicslice.
  }];
  let arguments = (ins
    PPHLO_Tensor:$operand,
    Variadic<PPHLO_ScalarIntTensor>:$start_indices,
    I64ElementsAttr:$slice_sizes
  );

  let results = (outs PPHLO_Tensor:$result);
}

def PPHLO_DynamicUpdateSliceOp: PPHLO_Op<"dynamic-update-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "update", "result"]>,
       AllShapesMatch<["operand", "result"]>]> {
  let summary = "Dynamic Update Slice operator";
  let description = [{
    DynamicUpdateSlice generates a result which is the value of the input array
    operand, with a slice update overwritten at start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice.
  }];
  let arguments = (ins
    PPHLO_Tensor:$operand,
    PPHLO_Tensor:$update,
    Variadic<PPHLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs PPHLO_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// pphlo control flow op definitions.
//===----------------------------------------------------------------------===//
def PPHLO_WhileOp : PPHLO_Op<"while", [
  RecursiveSideEffects, PPHLO_PairwiseSameOperandAndResultType,
  SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "While operator";
  let description = [{
    Returns the result of executing a body function until the cond body
        returns true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];
  let arguments = (ins Variadic<PPHLO_Tensor> : $args);

  let regions = (region SizedRegion<1> : $cond, SizedRegion<1> : $body);

  let results = (outs Variadic<PPHLO_Tensor>);
}

def PPHLO_IfOp : PPHLO_Op<"if", [
  RecursiveSideEffects,
  SingleBlockImplicitTerminator<"ReturnOp">
]> {
  let summary = "If operator";

  let description = [{
    Returns the result of executing either a true or false function depending on
        the result of a condition function.

    See https://www.tensorflow.org/xla/operation_semantics#conditional.
  }];

  let arguments = (ins PPHLO_IntTensor:$condition);

  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);

  let results = (outs Variadic<PPHLO_Tensor>);
}

#endif  // SPU_DIALECT_PPHLO_OPS
