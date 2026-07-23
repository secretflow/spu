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

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

#include "libspu/dialect/utils/assembly_format.h"

namespace mlir::spu::pphlo {

template <class... OpTypes>
void printSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                    OpTypes... types) {
  mlir::spu::printSameOperandsAndResultType(p, op,
                                            std::forward<OpTypes>(types)...);
}

template <class... OpTypes>
ParseResult parseSameOperandsAndResultType(OpAsmParser& parser,
                                           OpTypes&... types) {
  return mlir::spu::parseSameOperandsAndResultType(
      parser, std::forward<OpTypes&>(types)...);
}

// SliceRanges - Used to print multi-dimensional ranges for slice.
void printSliceRanges(OpAsmPrinter& p, Operation* op,
                      ArrayRef<int64_t> start_indices,
                      ArrayRef<int64_t> limit_indices,
                      ArrayRef<int64_t> strides);

ParseResult parseSliceRanges(OpAsmParser& parser,
                             DenseI64ArrayAttr& start_indices,
                             DenseI64ArrayAttr& limit_indices,
                             DenseI64ArrayAttr& strides);

// DotDimensionNumbers - Abbreviated printing using a ConvDimensionNumbers-
// inspired notation. batching_dims are skipped if empty.
//
// Generic:
//    dot_dimension_numbers = #stablehlo.dot<
//      lhs_batching_dimensions = [],
//      lhs_contracting_dimensions = [1],
//      rhs_batching_dimensions = [],
//      rhs_contracting_dimensions = [0]
//    >
//    dot_dimension_numbers = #stablehlo.dot<
//      lhs_batching_dimensions = [0],
//      lhs_contracting_dimensions = [2],
//      rhs_batching_dimensions = [0],
//      rhs_contracting_dimensions = [1]
//    >
//
// Custom:
//    contracting_dims = [1] x [0]
//    batching_dims = [0] x [0], contracting_dims = [2] x [1]
template <typename AttrTy>
void printDotDimensionNumbers(AsmPrinter& p, Operation* op, AttrTy target) {
  // Print two ArrayRef<int64_t> as `[...] x [...]`
  auto printLhsRhsDims = [&](ArrayRef<int64_t> lhs_dims,
                             ArrayRef<int64_t> rhs_dims) {
    DenseI64ArrayAttr::get(op->getContext(), lhs_dims).print(p);
    p << " x ";
    DenseI64ArrayAttr::get(op->getContext(), rhs_dims).print(p);
  };

  // Print the optional `batching_dims = [...] x [...]`.
  if (!target.getLhsBatchingDimensions().empty() ||
      !target.getRhsBatchingDimensions().empty()) {
    p << "batching_dims = ";
    printLhsRhsDims(target.getLhsBatchingDimensions(),
                    target.getRhsBatchingDimensions());
    p << ", ";
  }

  // Print the required `contracting_dims = [...] x [...]`.
  p << "contracting_dims = ";
  printLhsRhsDims(target.getLhsContractingDimensions(),
                  target.getRhsContractingDimensions());
}

template <typename AttrTy>
ParseResult parseDotDimensionNumbers(AsmParser& parser, AttrTy& target) {
  // Parse `[...] x [...]` into two DenseI64ArrayAttr attributes.
  auto parseLhsRhsDims = [&](DenseI64ArrayAttr& lhs_dims,
                             DenseI64ArrayAttr& rhs_dims) -> ParseResult {
    lhs_dims = mlir::dyn_cast_or_null<DenseI64ArrayAttr>(
        DenseI64ArrayAttr::parse(parser, Type{}));
    if (!lhs_dims) {
      return failure();
    }
    if (failed(parser.parseKeyword("x"))) {
      return failure();
    }
    rhs_dims = mlir::dyn_cast_or_null<DenseI64ArrayAttr>(
        DenseI64ArrayAttr::parse(parser, Type{}));
    if (!rhs_dims) {
      return failure();
    }
    return success();
  };

  // Parse the optional `batching_dims = [...] x [...]`.
  DenseI64ArrayAttr lhsBatchingDims;
  DenseI64ArrayAttr rhsBatchingDims;
  if (succeeded(parser.parseOptionalKeyword("batching_dims"))) {
    if (failed(parser.parseEqual()) ||
        failed(parseLhsRhsDims(lhsBatchingDims, rhsBatchingDims)) ||
        failed(parser.parseComma())) {
      return failure();
    }
  }

  // Parse the required `contracting_dims = [...] x [...]`.
  DenseI64ArrayAttr lhsContractingDims;
  DenseI64ArrayAttr rhsContractingDims;
  if (failed(parser.parseKeyword("contracting_dims")) ||
      failed(parser.parseEqual()) ||
      failed(parseLhsRhsDims(lhsContractingDims, rhsContractingDims))) {
    return failure();
  }

  target = AttrTy::get(
      parser.getBuilder().getContext(),
      lhsBatchingDims ? lhsBatchingDims.asArrayRef() : ArrayRef<int64_t>{},
      rhsBatchingDims ? rhsBatchingDims.asArrayRef() : ArrayRef<int64_t>{},
      lhsContractingDims.asArrayRef(), rhsContractingDims.asArrayRef());
  return success();
}

}  // namespace mlir::spu::pphlo
