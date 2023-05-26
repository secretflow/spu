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

namespace pphlo {

/// Lowers from HLO dialect to pphlo dialect
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToPPHloPass();

// Decompose comparison into lower ops when possible
std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeComparisonPass();

// Categorize a normal reduce into categorized reduce ops
std::unique_ptr<OperationPass<func::FuncOp>> createCategorizeReducePass();

// Lower UnrealizedConversionCastOp
std::unique_ptr<OperationPass<func::FuncOp>> createLowerConversionCastPass();

// Lower min/max
std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeMinMaxPass();

// Reduce truncation
std::unique_ptr<OperationPass<func::FuncOp>> createReduceTruncationPass();

// Lower mixed-type op
std::unique_ptr<OperationPass<func::FuncOp>> createLowerMixedTypeOpPass();

// Optimize MaxPooling layer
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeMaxPoolingPass();

// Optimize SelectOp
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSelectPass();

// Optimize sqrt(x) + very_small_const) -> sqrt(x + eps)
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSqrtPlusEps();

std::unique_ptr<OperationPass<func::FuncOp>> createExpandSecretGatherPass();

// Rewrite x/sqrt(x+eps) -> x*rsqrt(x+eps)
std::unique_ptr<OperationPass<func::FuncOp>> createRewriteDivSqrtPatterns();

// Optimize x/broadcast(y) into x*broadcast(1/y)
std::unique_ptr<OperationPass<func::FuncOp>>
createOptimizeDenominatorWithBroadcast();

} // namespace pphlo

} // namespace mlir
