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

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "libspu/dialect/ring/IR/assembly_format.h"
#include "libspu/dialect/ring/IR/type_helper.h"

// Put it here
#include "libspu/dialect/ring/IR/types.h"

namespace mlir::spu::ring::OpTrait {

template <typename ConcreteType>
class SameOperandsAndResultsSemanticRingType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameOperandsAndResultsSemanticRingType> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    auto base = getBaseType(op->getOperandTypes()[0]);

    auto numOperands = op->getNumOperands();
    for (auto idx : llvm::seq<unsigned int>(1, numOperands)) {
      if (base != getBaseType(op->getOperandTypes()[idx])) {
        return op->emitOpError()
               << "requires the same type for operand at index " << idx;
      }
    }

    auto numResults = op->getNumResults();
    for (auto idx : llvm::seq<unsigned int>(1, numResults)) {
      if (base != getBaseType(op->getResultTypes()[idx])) {
        return op->emitOpError()
               << "requires the same type for result at index " << idx;
      }
    }

    return success();
  }
};

template <typename ConcreteType>
class SameOperandsSemanticRingType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameOperandsSemanticRingType> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    auto base = getBaseType(op->getOperandTypes()[0]);

    auto numOperands = op->getNumOperands();
    for (auto idx : llvm::seq<unsigned int>(1, numOperands)) {
      if (base != getBaseType(op->getOperandTypes()[idx])) {
        return op->emitOpError()
               << "requires the same type for operand at index " << idx;
      }
    }

    return success();
  }
};

template <typename ConcreteType>
class SameOperandsAndResultsVisibilityType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameOperandsAndResultsVisibilityType> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    auto base_is_public = isPublic(op->getOperandTypes()[0]);

    auto numOperands = op->getNumOperands();
    for (auto idx : llvm::seq<unsigned int>(1, numOperands)) {
      if (base_is_public != isPublic(op->getOperandTypes()[idx])) {
        return op->emitOpError()
               << "requires the same visibility for operand at index " << idx;
      }
    }

    auto numResults = op->getNumResults();
    for (auto idx : llvm::seq<unsigned int>(1, numResults)) {
      if (base_is_public != isPublic(op->getResultTypes()[idx])) {
        return op->emitOpError()
               << "requires the same visibiity for result at index " << idx;
      }
    }

    return success();
  }
};

}  // namespace mlir::spu::ring::OpTrait

#define GET_OP_CLASSES
#include "libspu/dialect/ring/IR/ops.h.inc"
