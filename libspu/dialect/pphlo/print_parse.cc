// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/TypeUtilities.h"

#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

namespace {

void createArgs(ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                ArrayRef<Type> types,
                SmallVector<OpAsmParser::Argument>& args) {
  for (auto argAndType : llvm::zip(operands, types)) {
    auto& arg = args.emplace_back();
    arg.ssaName = std::get<0>(argAndType);
    arg.type = std::get<1>(argAndType);
  }
}

bool hasSameOperandAndResultTypes(Operation& op) {
  Type expected;
  if (op.getNumResults() != 0) {
    expected = op.getResult(0).getType();
  }
  if (op.getNumOperands() != 0) {
    expected = op.getOperand(0).getType();
  }
  if (!expected) {
    return false;
  }

  auto typeMatch = [&](Type actual) { return actual == expected; };
  return llvm::all_of(op.getOperandTypes(), typeMatch) &&
         llvm::all_of(op.getResultTypes(), typeMatch);
}

bool isEligibleForCompactPrint(ReduceOp op) {
  // Check E1.
  auto& block = op.getBody().front();
  if (!hasSingleElement(block.without_terminator())) {
    return false;
  }

  Operation& innerOp = *block.begin();

  // Check E2.
  if (innerOp.getDialect() != op->getDialect()) {
    return false;
  }

  if (innerOp.getNumOperands() != 2 ||
      !innerOp.hasTrait<mlir::OpTrait::OneResult>() ||
      !hasSameOperandAndResultTypes(innerOp) ||
      !innerOp.hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOp.hasTrait<mlir::OpTrait::ZeroRegions>()) {
    return false;
  }

  // Check E3.
  if (op.getInputs().empty()) {
    return false;
  }

  auto elemType =
      mlir::dyn_cast<ShapedType>(op.getInputs()[0].getType()).getElementType();
  auto expectedInnerOpType = RankedTensorType::get(/*shape=*/{}, elemType);
  if (innerOp.getOperands()[0].getType() != expectedInnerOpType) {
    return false;
  }

  // Check E4.
  if (!llvm::equal(block.getArguments(), innerOp.getOperands())) {
    return false;
  }

  // Check E5.
  auto retOp = dyn_cast<ReturnOp>(block.getTerminator());
  if (!retOp) {
    return false;
  }

  return llvm::equal(innerOp.getResults(), retOp.getOperands());
}

}  // namespace

// ----------------------- ConstantOp ----------------------------------//
ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the generic form.
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseRParen()) {
      return failure();
    }
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    if (parser.parseColon() || parser.parseLParen() || parser.parseRParen() ||
        parser.parseArrow()) {
      return failure();
    }
    Type resultTy;
    if (parser.parseType(resultTy)) {
      return failure();
    }
    result.addTypes(resultTy);
    return success();
  }

  ElementsAttr valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (parser.parseCustomAttributeWithFallback(valueAttr, Type{}, "value",
                                              result.attributes)) {
    return failure();
  }
  result.addTypes(valueAttr.getType());
  return success();
}

/// Print a `constant` op.
///
/// op ::= attr-dict $value
///
/// When the `value` and `output` have different type, it just uses the default
/// operator assembly format as a fallback.
void ConstantOp::print(::mlir::OpAsmPrinter& p) {
  // If not all types are the same, use generic form.
  if (getValue().getType() != getType()) {
    p.printGenericOp(getOperation(), /*printOpName=*/false);
    return;
  }

  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << ' ';
  p.printStrippedAttrOrType(getValueAttr());
}

// ----------------------- ReduceOp ----------------------------------//
void ReduceOp::print(OpAsmPrinter& p) {
  {
    // Print the pairs of operands under the form:
    //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
    StringRef comma = "";
    int numOperandPairs = getNumOperands() / 2;
    for (int opId : llvm::seq<int>(0, numOperandPairs)) {
      p << comma << "(" << getOperand(opId)
        << " init: " << getOperand(opId + numOperandPairs) << ")";
      comma = ", ";
    }
  }

  // If the reduce-op is eligible for compact printing, we emit the one-liner:
  // pphlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // Note: We are not printing the function type of reduction operation. We
  // have some simplifying assumptions (refer to IsEligibleForCompactPrint::E3)
  // to derive the type from that of reduce-op.
  if (isEligibleForCompactPrint(*this)) {
    Operation& innerOp = getBody().front().front();
    p << " applies ";
    printEscapedString(innerOp.getName().getStringRef(), p.getStream());

    p << " across dimensions = [";
    llvm::interleaveComma(getDimensions(), p);
    p << "]";
    p.printOptionalAttrDict(getOperation()->getAttrs(), {"dimensions"});
    p << " : ";
    p.printFunctionalType(*this);
  } else {
    p << " across dimensions = [";
    llvm::interleaveComma(getDimensions(), p);
    p << "]";
    p.printOptionalAttrDict(getOperation()->getAttrs(), {"dimensions"});
    p << " : ";
    p.printFunctionalType(*this);
    p.printNewline();
    p << " reducer";
    {
      // Print the pairs of block operands under the form:
      //   (%arg0_elt, %arg0_acc) (%arg1_elt, %arg1_acc):
      Block& reducer = getBody().front();
      int numOperandPairs = getNumOperands() / 2;
      for (int opId : llvm::seq<int>(0, numOperandPairs)) {
        p << "(";
        p.printRegionArgument(reducer.getArgument(opId));
        p << ", ";
        p.printRegionArgument(reducer.getArgument(opId + numOperandPairs));
        p << ") ";
      }
    }
    p << ' ';
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  }
}

ParseResult ReduceOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands of reduce-op, this is a list of pair under the form:
  //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
  // Each input to reduce is paired with its init value, even though in memory
  // they are stored with the input first and the init values after.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> initOperands;
  do {
    (void)parser.parseOptionalComma();
    if (parser.parseOptionalLParen()) {
      break;
    }
    OpAsmParser::UnresolvedOperand operand;
    OpAsmParser::UnresolvedOperand initOperand;
    if (parser.parseOperand(operand) || parser.parseKeyword("init") ||
        parser.parseColon() || parser.parseOperand(initOperand) ||
        parser.parseRParen()) {
      return failure();
    }
    operands.push_back(operand);
    initOperands.push_back(initOperand);
  } while (true);
  operands.append(initOperands);

  // Check if we are parsing the compact version of reduce-op:
  // pphlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // else parse the "region-based" variant.
  if (failed(parser.parseOptionalKeyword("applies"))) {
    // Parse the inner-op dimensions, reduce-op's function-type and
    // optional location.
    SmallVector<int64_t> dimensions;
    auto parseDim = [&]() -> ParseResult {
      if (parser.parseInteger(dimensions.emplace_back())) {
        return failure();
      }
      return success();
    };

    FunctionType reduceOpFnType;
    if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
        parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       parseDim) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(reduceOpFnType) ||
        parser.parseKeyword("reducer")) {
      return failure();
    }
    OpBuilder builder(parser.getBuilder().getContext());
    result.addAttribute("dimensions", builder.getDenseI64ArrayAttr(dimensions));

    // Parse the "reducer" region now.
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerOperands;
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerInitOperands;
    SmallVector<Type, 2> reducerTypes;
    SmallVector<Type, 2> reducerInitTypes;
    SmallVector<std::optional<Location>, 2> reducerLocs;
    SmallVector<std::optional<Location>, 2> reducerInitLocs;
    auto parseBlockOperand =
        [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
            SmallVectorImpl<Type>& types,
            SmallVectorImpl<std::optional<Location>>& locs) -> ParseResult {
      OpAsmParser::UnresolvedOperand operand;
      Type type;
      std::optional<Location> loc;
      if (parser.parseOperand(operand, /*allowResultNumber=*/false) ||
          parser.parseColon() || parser.parseType(type) ||
          parser.parseOptionalLocationSpecifier(loc)) {
        return failure();
      }
      operands.push_back(operand);
      types.push_back(type);
      locs.push_back(loc);
      return success();
    };
    do {
      if (failed(parser.parseOptionalLParen())) {
        break;
      }
      if (parseBlockOperand(reducerOperands, reducerTypes, reducerLocs) ||
          parser.parseComma() ||
          parseBlockOperand(reducerInitOperands, reducerInitTypes,
                            reducerInitLocs) ||
          parser.parseRParen()) {
        return failure();
      }
    } while (true);
    reducerOperands.append(reducerInitOperands);
    reducerTypes.append(reducerInitTypes);
    reducerLocs.append(reducerInitLocs);
    result.addTypes(reduceOpFnType.getResults());
    SmallVector<OpAsmParser::Argument> reducerArgs;
    createArgs(reducerOperands, reducerTypes, reducerArgs);

    // Derive the SSA-values for reduce-op's operands and parse the region, and
    // the optional trailing location.
    std::optional<Location> trailingLoc;
    if (parser.resolveOperands(operands, reduceOpFnType.getInputs(), loc,
                               result.operands) ||
        parser.parseRegion(*result.addRegion(), reducerArgs)) {
      return failure();
    }
    // Set the individual block arguments.
    for (auto argAndLoc : llvm::zip(
             result.regions.front()->front().getArguments(), reducerLocs)) {
      if (std::get<1>(argAndLoc)) {
        std::get<0>(argAndLoc).setLoc(std::get<1>(argAndLoc).value());
      }
    }
    result.location = trailingLoc.value_or(currLocation);
    return success();
  }

  // Parse the inner-op name and check if the contract on inner-op
  // mentioned in "isEligibleForCompactPrint::E2" for pretty-printing is met.
  FailureOr<OperationName> innerOpNameInfo = parser.parseCustomOperationName();
  if (failed(innerOpNameInfo)) {
    return failure();
  }

  StringRef innerOpName = innerOpNameInfo->getStringRef();
  Dialect* innerOpDialect = innerOpNameInfo->getDialect();
  if ((innerOpDialect == nullptr) ||
      !innerOpDialect->getNamespace().equals("pphlo") ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::NOperands<2>::Impl>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::OneResult>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::ZeroRegions>()) {
    parser.emitError(loc,
                     "expected the inner-op to be a commutative binary-op from "
                     "pphlo dialect, zero region, producing single result");
    return failure();
  }

  // Parse the inner-op dimensions, reduce-op's function-type and
  // optional location.
  SmallVector<int64_t> dimensions;
  auto parseDim = [&]() -> ParseResult {
    if (parser.parseInteger(dimensions.emplace_back())) {
      return failure();
    }
    return success();
  };

  std::optional<Location> explicitLoc;
  FunctionType reduceOpFnType;
  if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseDim) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(reduceOpFnType) ||
      parser.parseOptionalLocationSpecifier(explicitLoc)) {
    return failure();
  }

  if (!reduceOpFnType || reduceOpFnType.getInputs().empty()) {
    if (!reduceOpFnType) {
      return parser.emitError(loc, "expected function type");
    }
    return parser.emitError(loc,
                            "input types missing in reduce-op function type");
  }

  // If location of reduce-op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location reduceOpLoc = explicitLoc.value_or(currLocation);

  // Derive the SSA-values for reduce-op's operands.
  if (parser.resolveOperands(operands, reduceOpFnType.getInputs(), loc,
                             result.operands)) {
    return failure();
  }

  // Derive the type of inner-op from that of reduce-op's input operand.
  auto innerOpType = RankedTensorType::get(
      /*shape=*/{}, getElementTypeOrSelf(reduceOpFnType.getInput(0)));

  // Add a region for reduce-op.
  Region& region = *result.addRegion();

  // Create a basic-block inside reduce-op's region.
  Block& block = region.emplaceBlock();
  auto lhs = block.addArgument(innerOpType, reduceOpLoc);
  auto rhs = block.addArgument(innerOpType, reduceOpLoc);

  // Create and insert an "inner-op" operation in the block.
  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToStart(&block);

  OperationState innerOpState(reduceOpLoc, innerOpName);
  innerOpState.operands.push_back(lhs);
  innerOpState.operands.push_back(rhs);
  innerOpState.addTypes(innerOpType);

  Operation* innerOp = builder.create(innerOpState);

  // Insert a return statement in the block returning the inner-op's result.
  builder.create<ReturnOp>(innerOp->getLoc(), innerOp->getResults());

  // Populate the reduce-op operation-state with result-type, location, and
  // dimension attribute.
  result.addTypes(reduceOpFnType.getResults());
  result.location = innerOp->getLoc();
  result.addAttribute("dimensions", builder.getDenseI64ArrayAttr(dimensions));
  return success();
}

// ----------------------- WhileOp ----------------------------------//

/// Print a `while` op.
///
/// op ::= `stablehlo.while` `(` assignment-list `)` `:` types attribute-dict
///         `cond` region
///         `do` region
/// assignment-list ::= assignment | assignment `,` assignment-list
/// assignment ::= ssa-value `=` ssa-value
void WhileOp::print(OpAsmPrinter& p) {
  p << '(';
  llvm::interleaveComma(
      llvm::zip(SingleBlock::getBody()->getArguments(), getOperands()), p,
      [&](auto zip) {
        p.printOperand(std::get<0>(zip));
        p << " = ";
        p.printOperand(std::get<1>(zip));
      });
  p << ")";
  if (getNumOperands() != 0u) {
    p << " : ";
    llvm::interleaveComma(getOperandTypes(), p);
  }
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p.printNewline();
  p << " cond ";
  p.printRegion(getRegion(0), /*printEntryBlockArgs=*/false);
  p << " do ";
  p.printRegion(getRegion(1), /*printEntryBlockArgs=*/false);
}

ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse the operands of the while: these are of the form:
  //   %iter_arg = %init_val
  // where %iter_arg is the name of the block argument in the cond/body blocks
  // and %init_val is the actual operand.
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgs;
  if (parser.parseLParen()) {
    return failure();
  }
  do {
    if (succeeded(parser.parseOptionalRParen())) {
      break;
    }
    OpAsmParser::UnresolvedOperand operand;
    OpAsmParser::UnresolvedOperand iterArg;
    if (parser.parseOperand(iterArg) || parser.parseEqual() ||
        parser.parseOperand(operand)) {
      return failure();
    }
    iterArgs.push_back(iterArg);
    operands.push_back(operand);
    if (succeeded(parser.parseOptionalRParen())) {
      break;
    }
    if (failed(parser.parseComma())) {
      return failure();
    }
  } while (true);
  if (!operands.empty()) {
    if (parser.parseColon() || parser.parseTypeList(result.types)) {
      return failure();
    }
  }
  SmallVector<OpAsmParser::Argument> args;
  createArgs(iterArgs, result.types, args);
  if (parser.resolveOperands(operands, result.types, loc, result.operands) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseKeyword("cond") ||
      parser.parseRegion(*result.addRegion(), args) ||
      parser.parseKeyword("do") ||
      parser.parseRegion(*result.addRegion(), args)) {
    return failure();
  }
  return success();
}

}  // namespace mlir::spu::pphlo
