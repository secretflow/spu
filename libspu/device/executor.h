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

#include <functional>
#include <shared_mutex>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::device {

//
class SymbolScope final {
  // The parent region, null if this region is isolated from above.
  SymbolScope *parent_;

  // Local symbols inside this value.
  mutable std::shared_mutex mu_;
  llvm::DenseMap<mlir::Value, spu::Value> symbols_;

 public:
  explicit SymbolScope(SymbolScope *parent = nullptr) : parent_(parent) {}

  // return true if this is the root scope.
  bool isRoot() const { return parent_ == nullptr; }

  //
  bool hasValue(mlir::Value key) const;
  bool hasValues(mlir::OperandRange keys) const;
  bool hasValues(llvm::ArrayRef<mlir::Value> keys) const;
  spu::Value lookupValue(mlir::Value key) const;
  void addValue(::mlir::Value key, const spu::Value &val);
  void addValue(::mlir::Value key, spu::Value &&val);

 protected:
  bool hasValueUnsafe(mlir::Value key) const;
};

// This class encapsulate execution states used during the evaluation.
struct ExecutionOptions {
  bool do_type_check = false;
  bool do_log_execution = false;
  bool do_parallel = false;
};

class OpExecutor {
 public:
  virtual ~OpExecutor() = default;

  //
  virtual void checkType(mlir::Type mlir_type, const spu::Value &v) const = 0;

  // return true if the operation has a corresponding kernel.
  virtual bool hasKernel(mlir::Operation &op) const = 0;

  // run a kernel in a given region.
  virtual void runKernelImpl(SPUContext *sctx, SymbolScope *sscope,
                             mlir::Operation &op,
                             const ExecutionOptions &opts) = 0;

  void runKernel(SPUContext *sctx, SymbolScope *sscope, mlir::Operation &op,
                 const ExecutionOptions &opts = {}) {
    return runKernelImpl(sctx, sscope, op, opts);
  }
};

std::vector<spu::Value> runRegion(OpExecutor *executor, SPUContext *sctx,
                                  SymbolScope *parent_scope,
                                  mlir::Region &region,
                                  absl::Span<spu::Value const> params,
                                  const ExecutionOptions &opts = {});

std::vector<spu::Value> runBlock(OpExecutor *executor, SPUContext *sctx,
                                 SymbolScope *symbols, mlir::Block &block,
                                 absl::Span<spu::Value const> params,
                                 const ExecutionOptions &opts);

std::vector<spu::Value> runBlockParallel(OpExecutor *executor, SPUContext *sctx,
                                         SymbolScope *symbols,
                                         mlir::Block &block,
                                         absl::Span<spu::Value const> params,
                                         const ExecutionOptions &opts);

}  // namespace spu::device
