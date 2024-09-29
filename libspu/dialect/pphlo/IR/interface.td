// Copyright 2023 Ant Group Co., Ltd.
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

#ifndef SPU_DIALECT_PPHLO_INTERFACE
#define SPU_DIALECT_PPHLO_INTERFACE

include "mlir/IR/OpBase.td"

def PPHloTypeInterface : TypeInterface<"PPHloTypeInterface"> {
  let cppNamespace = "mlir::spu::pphlo";

  let description = [{
    Interface for encapsulating the common properties of pphlo types.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Get bit-width of the type.",
      /*retTy=*/"unsigned",
      /*methodName=*/"getWidth"
    >
  ];
}

def InferFxpRangeInterface : OpInterface<"InferFxpRangeInterface"> {
  let description = [{
    Allows operations to participate in range analysis for values by
    providing a method that allows them to specify lower and upper bounds on their
    result(s) given lower and upper bounds on their input(s) if known.
  }];
  let cppNamespace = "mlir::spu::pphlo";

  let methods = [
    InterfaceMethod<[{
      Infer the bounds on the results of this op given the bounds on its arguments.
      For each result value or block argument (that isn't a branch argument,
      since the dataflow analysis handles those case), the method should call
      `setValueRange` with that `Value` as an argument. When implemented,
      `setValueRange` should be called on all result values for the operation.
    }],
    /*retTy=*/"void",
    /*methodName=*/"inferResultRanges",
    /*args=*/(ins "::llvm::ArrayRef<::mlir::ConstantIntRanges>":$argRanges,
                  "::mlir::SetIntRangeFn":$setResultRanges),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      fxprange::detail::defaultInferResultRanges($_op,
                                                 argRanges,
                                                 setResultRanges);
    }]>,

    InterfaceMethod<[{
      Infer the bounds on the results of this op given the lattice representation
      of the bounds for its arguments. For each result value or block argument
      (that isn't a branch argument, since the dataflow analysis handles
      those case), the method should call `setValueRange` with that `Value`
      as an argument. When implemented, `setValueRange` should be called on
      all result values for the operation.

      This method allows for more precise implementations when operations
      want to reason about inputs which may be undefined during the analysis.
    }],
    /*retTy=*/"void",
    /*methodName=*/"inferResultRangesFromOptional",
    /*args=*/(ins "::llvm::ArrayRef<::mlir::IntegerValueRange>":$argRanges,
                  "::mlir::SetIntLatticeFn":$setResultRanges),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      fxprange::detail::defaultInferResultRangesFromOptional($_op,
                                                             argRanges,
                                                             setResultRanges);
    }]>
  ];
}

#endif
