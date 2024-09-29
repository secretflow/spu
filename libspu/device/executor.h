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

#include <shared_mutex>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "libspu/core/context.h"
#include "libspu/core/memref.h"

namespace spu::device {

//
class SymbolScope final {
  // The parent region, null if this region is isolated from above.
  SymbolScope *parent_;

  // Local symbols inside this value.
  mutable std::shared_mutex mu_;
  llvm::DenseMap<mlir::Value, spu::MemRef> symbols_;

 public:
  explicit SymbolScope(SymbolScope *parent = nullptr) : parent_(parent) {}

  // return true if this is the root scope.
  bool isRoot() const { return parent_ == nullptr; }

  //
  bool hasValue(mlir::Value key) const;
  bool hasValues(mlir::OperandRange keys) const;
  bool hasValues(llvm::ArrayRef<mlir::Value> keys) const;
  spu::MemRef lookupValue(mlir::Value key) const;
  void addValue(::mlir::Value key, const spu::MemRef &val);
  void addValue(::mlir::Value key, spu::MemRef &&val);
  void removeValue(::mlir::Value key);

 protected:
  bool hasValueUnsafe(mlir::Value key) const;
};

// This class encapsulate execution states used during the evaluation.
struct ExecutionOptions {
  bool do_type_check = false;
  bool do_log_execution = false;
  bool do_parallel = false;
  uint64_t concurrency = 0;
};

class OpExecutor {
 public:
  virtual ~OpExecutor() = default;

  // run a kernel in a given region.
  virtual void runKernelImpl(SPUContext *sctx, SymbolScope *sscope,
                             mlir::Operation &op,
                             const ExecutionOptions &opts) = 0;

  void runKernel(SPUContext *sctx, SymbolScope *sscope, mlir::Operation &op,
                 const ExecutionOptions &opts = {}) {
    return runKernelImpl(sctx, sscope, op, opts);
  }

  using handler_t = std::function<bool(SPUContext *sctx, mlir::Operation *op,
                                       absl::Span<const spu::MemRef> inputs)>;
  void setExtraIntrinsicHandler(handler_t handler) {
    extra_handler_ = std::move(handler);
  }

  const std::optional<handler_t> &getExtraIntrinsicHandler() const {
    return extra_handler_;
  }

 private:
  std::optional<handler_t> extra_handler_;
};

std::vector<spu::MemRef> runRegion(OpExecutor *executor, SPUContext *sctx,
                                   SymbolScope *parent_scope,
                                   mlir::Region &region,
                                   absl::Span<spu::MemRef const> params,
                                   const ExecutionOptions &opts = {});

std::vector<spu::MemRef> runBlock(OpExecutor *executor, SPUContext *sctx,
                                  SymbolScope *symbols, mlir::Block &block,
                                  absl::Span<spu::MemRef const> params,
                                  const ExecutionOptions &opts);

std::vector<spu::MemRef> runBlockParallel(OpExecutor *executor,
                                          SPUContext *sctx,
                                          SymbolScope *symbols,
                                          mlir::Block &block,
                                          absl::Span<spu::MemRef const> params,
                                          const ExecutionOptions &opts);

}  // namespace spu::device
