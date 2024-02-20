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

#include "libspu/dialect/pphlo/AssemblyFormat.h"

namespace mlir::spu::pphlo {

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
  if (auto fnType = type.dyn_cast<FunctionType>()) {
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

void printSliceRanges(OpAsmPrinter& p, Operation* op,
                      ArrayRef<int64_t> start_indices,
                      ArrayRef<int64_t> limit_indices,
                      ArrayRef<int64_t> strides) {
  p << "[";
  // Let's be safe if we're printing invalid IR somehow: this can't be parsed
  // back!
  if (start_indices.size() != limit_indices.size() ||
      start_indices.size() != strides.size()) {
    p << "start_indices: ";
    llvm::interleaveComma(start_indices, p);
    p << ", limit_indices: ";
    llvm::interleaveComma(limit_indices, p);
    p << ", strides: ";
    llvm::interleaveComma(strides, p);
    p << "]";
    return;
  }

  llvm::interleaveComma(llvm::zip(start_indices, limit_indices, strides), p,
                        [&](std::tuple<int64_t, int64_t, int64_t> pack) {
                          auto [start, limit, stride] = pack;
                          p << start << ":" << stride << ":" << limit;
                        });
  p << "]";
}

ParseResult parseSliceRanges(OpAsmParser& parser,
                             DenseI64ArrayAttr& start_indices,
                             DenseI64ArrayAttr& limit_indices,
                             DenseI64ArrayAttr& strides) {
  if (parser.parseLSquare()) {
    return failure();
  }
  // Parse groups of comma-separated: `start`:`stride`:`limit`
  SmallVector<int64_t> start;
  SmallVector<int64_t> limit;
  SmallVector<int64_t> stride;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      start.emplace_back();
      stride.emplace_back();
      limit.emplace_back();
      if (parser.parseInteger(start.back()) || parser.parseColon() ||
          parser.parseInteger(stride.back()) || parser.parseColon() ||
          parser.parseInteger(limit.back())) {
        return failure();
      }
      if (succeeded(parser.parseOptionalRSquare())) {
        break;
      }
      if (failed(parser.parseComma())) {
        return failure();
      }
    } while (true);
  }

  start_indices = parser.getBuilder().getDenseI64ArrayAttr(start);
  limit_indices = parser.getBuilder().getDenseI64ArrayAttr(limit);
  strides = parser.getBuilder().getDenseI64ArrayAttr(stride);

  return success();
}

}  // namespace mlir::spu::pphlo
