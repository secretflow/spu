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

namespace mlir {

namespace func {
class FuncOp;
}

class ModuleOp;
class Pass;
template <typename T>
class OperationPass;

namespace pphlo {

/// Lowers from HLO dialect to pphlo dialect with cli io_vis
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToPPHloPass();

/// Lower from HLO dialect to pphlo dialect with a json io_vis string
std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeToPPHloPass(const std::string &io_visibility_json);

// Decompose comparison into lower ops when possible
std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeComparisonPass();

// Decompose sqrt into lower ops
std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeSqrtPass();

// Categorize a normal reduce into categorized reduce ops
std::unique_ptr<OperationPass<func::FuncOp>> createCategorizeReducePass();

// Lower UnrealizedConversionCastOp
std::unique_ptr<OperationPass<func::FuncOp>> createLowerConversionCastPass();

} // namespace pphlo
} // namespace mlir
