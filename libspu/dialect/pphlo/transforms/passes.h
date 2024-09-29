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

#pragma once

#include <memory>

namespace spu::compiler {
class CompilationContext;
}

namespace mlir {
namespace func {
class FuncOp;
}

class ModuleOp;
class Pass;
template <typename T>
class OperationPass;

namespace spu ::pphlo {

/// Lowers from HLO dialect to pphlo dialect
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToPPHloPass();

// Lower UnrealizedConversionCastOp
std::unique_ptr<OperationPass<func::FuncOp>> createLowerConversionCastPass();

// Lower high-level ops into basic ops
std::unique_ptr<OperationPass<func::FuncOp>> createGeneralDecomposeOps();
std::unique_ptr<OperationPass<func::FuncOp>> createSecretDecomposeOps();

// Reduce truncation
std::unique_ptr<OperationPass<func::FuncOp>> createReduceTruncationPass();

// Lower mixed-type op
std::unique_ptr<OperationPass<func::FuncOp>> createLowerMixedTypeOpPass();

// Optimize SelectOp
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSelectPass();

// Optimize sqrt(x) + very_small_const) -> sqrt(x + eps)
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSqrtPlusEps();

// Rewrite x/sqrt(x+eps) -> x*rsqrt(x+eps)
std::unique_ptr<OperationPass<func::FuncOp>> createRewriteDivSqrtPatterns();

// Optimize x/broadcast(y) into x*broadcast(1/y)
std::unique_ptr<OperationPass<func::FuncOp>>
createOptimizeDenominatorWithBroadcast();

// Lower sort with simple comprators to simple sort
std::unique_ptr<OperationPass<func::FuncOp>> createSortLowering();

// Push convert later
std::unique_ptr<OperationPass<func::FuncOp>> createConvertPushDownPass();

// Convert IR to fixedpoint
std::unique_ptr<OperationPass<func::FuncOp>> createExpandFixedPointApprox();
std::unique_ptr<OperationPass<func::FuncOp>> createLowerSecretFloatToFxp();
std::unique_ptr<OperationPass<func::FuncOp>> createLowerPPHloFloatInputs();

// Convert partial sort to topk
std::unique_ptr<OperationPass<func::FuncOp>> createPartialSortToTopK();

// Inline secret if/case
std::unique_ptr<OperationPass<func::FuncOp>> createInlineSecretControlFlow();

// DotGeneral to dot+concat
std::unique_ptr<OperationPass<func::FuncOp>> createDotGeneralToDot();

// Convert signbit pattern to SignOp
std::unique_ptr<OperationPass<func::FuncOp>> createRewriteSignbitPatterns();

// Create range based optimization
std::unique_ptr<OperationPass<func::FuncOp>> createRangeOptimization();

// Legalize to scf
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToSCF();

// Legalize to tensor
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToTensor();

// Legalize to linalg
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToLinalg();

// Legalize to arith
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToArith();

}  // namespace spu::pphlo

}  // namespace mlir
