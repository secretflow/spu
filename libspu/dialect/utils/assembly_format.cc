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

#include "libspu/dialect/utils/assembly_format.h"

namespace mlir::spu {

namespace {

ParseResult assignFromFunctionType(OpAsmParser& parser, llvm::SMLoc loc,
                                   ArrayRef<Type*> operands, Type& result,
                                   FunctionType& fn_type) {
  assert(fn_type);
  if (fn_type.getInputs().size() != operands.size()) {
    return parser.emitError(loc)
           << operands.size() << " operands present, but expected "
           << fn_type.getInputs().size();
  }

  // Set operand types to function input types
  for (auto [operand, input] : llvm::zip(operands, fn_type.getInputs())) {
    *operand = input;
  }

  // Set result type
  if (fn_type.getResults().size() != 1) {
    return parser.emitError(loc, "expected single output");
  }
  result = fn_type.getResults()[0];

  return success();
}

}  // namespace

namespace detail {

void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                        TypeRange operands, Type result) {
  // Handle zero operand types `() -> a` prints `a`
  if (operands.empty()) {
    p.printType(result);
    return;
  }
  // Handle all same type `(a,a,...) -> a` prints `a`
  bool allSameType =
      llvm::all_of(operands, [&result](auto t) { return t == result; });
  if (allSameType) {
    p.printType(result);
    return;
  }
  // Fall back to generic
  p.printFunctionalType(op);
}

ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                               ArrayRef<Type*> operands,
                                               Type& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  Type type;
  if (parser.parseType(type)) {
    return failure();
  }

  // Handle if function type, all operand types did not match result type.
  if (auto fnType = mlir::dyn_cast<FunctionType>(type)) {
    return assignFromFunctionType(parser, loc, operands, result, fnType);
  }

  // Handle bare types. ` : type` indicating all input/output types match.
  for (Type* t : operands) {
    *t = type;
  }
  result = type;
  return success();
}

}  // namespace detail

void printCustomCallTargetImpl(AsmPrinter& p, StringAttr target) {
  p.printSymbolName(target.getValue());
}

ParseResult parseCustomCallTargetImpl(AsmParser& parser, StringAttr& target) {
  return parser.parseSymbolName(target);
}

}  // namespace mlir::spu
