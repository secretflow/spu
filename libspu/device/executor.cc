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

#include "libspu/device/executor.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/core/value.h"

namespace spu::device {

spu::Value SymbolScope::lookupValue(mlir::Value key) const {
  {
    std::shared_lock<std::shared_mutex> lk(mu_);
    auto itr = symbols_.find(key);

    if (itr != symbols_.end()) {
      return itr->second;
    }
  }

  if (parent_ != nullptr) {
    return parent_->lookupValue(key);
  }

  // Somehow cannot find this value on stack, print a reasonable error
  SPDLOG_ERROR("Should not be here, symbol not found");
  SPU_THROW("TODO: add more details");
  // SPU_THROW("Try to get a non-exist value, defined at {} ",
  //            mlirObjectToString(*v.getDefiningOp()));
}

bool SymbolScope::hasValueUnsafe(mlir::Value key) const {
  auto itr = symbols_.find(key);

  if (itr != symbols_.end()) {
    return true;
  }

  if (parent_ != nullptr) {
    return parent_->hasValue(key);
  }

  return false;
}

bool SymbolScope::hasValues(mlir::OperandRange keys) const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  return std::all_of(keys.begin(), keys.end(), [this](const mlir::Value &key) {
    return hasValueUnsafe(key);
  });
}

bool SymbolScope::hasValues(llvm::ArrayRef<mlir::Value> keys) const {
  if (keys.empty()) {
    return true;
  }
  std::shared_lock<std::shared_mutex> lk(mu_);
  return std::all_of(keys.begin(), keys.end(), [this](const mlir::Value &key) {
    return hasValueUnsafe(key);
  });
}

bool SymbolScope::hasValue(mlir::Value key) const {
  std::shared_lock<std::shared_mutex> lk(mu_);
  return hasValueUnsafe(key);
}

void SymbolScope::addValue(mlir::Value key, const spu::Value &val) {
  std::lock_guard<std::shared_mutex> lk(mu_);
  symbols_[key] = val;
}

void SymbolScope::addValue(mlir::Value key, spu::Value &&val) {
  std::lock_guard<std::shared_mutex> lk(mu_);
  symbols_[key] = std::move(val);
}

std::vector<spu::Value> runRegion(OpExecutor *executor,                 //
                                  SPUContext *sctx,                     //
                                  SymbolScope *parent_scope,            //
                                  mlir::Region &region,                 //
                                  absl::Span<spu::Value const> params,  //
                                  const ExecutionOptions &opts) {
  SPU_ENFORCE(region.getNumArguments() == params.size(),
              "region requires {} arguments while got number of params {}",
              region.getRegionNumber(), params.size());

  // create a new scope for this region.
  SymbolScope sscope(parent_scope);

  // inject the parameters to region's symbol table.
  for (const auto &blkarg : region.getArguments()) {
    sscope.addValue(blkarg, params[blkarg.getArgNumber()]);
  }

  SPU_ENFORCE(region.hasOneBlock());
  if (opts.do_parallel) {
    return runBlockParallel(executor, sctx, &sscope, region.front(), params,
                            opts);
  } else {
    return runBlock(executor, sctx, &sscope, region.front(), params, opts);
  }
}

std::vector<spu::Value> runBlock(OpExecutor *executor, SPUContext *sctx,
                                 SymbolScope *symbols, mlir::Block &block,
                                 absl::Span<spu::Value const> params,
                                 const ExecutionOptions &opts) {
  for (auto &op : block.without_terminator()) {
    executor->runKernel(sctx, symbols, op, opts);
  }

  if (auto *termOp = block.getTerminator()) {
    // TODO: enforce ReturnLike
    std::vector<spu::Value> results;
    results.reserve(termOp->getNumOperands());
    for (const auto operand : termOp->getOperands()) {
      results.emplace_back(symbols->lookupValue(operand));
    }
    return results;
  }

  // No terminator
  SPU_THROW("Should not be here");
}

struct SymbolTableEvent {
  std::condition_variable cv;
  std::mutex mutex;
};

class OpExecTask final {
  std::unique_ptr<SPUContext> sctx_ = nullptr;
  // here we assume executor is thread-safe (stateless)
  OpExecutor *executor_ = nullptr;
  SymbolScope *sscope_ = nullptr;
  mlir::Operation *op_ = nullptr;
  SymbolTableEvent *event_ = nullptr;
  llvm::SmallVector<mlir::Value> extra_dependencies_;

 public:
  OpExecTask() = default;
  explicit OpExecTask(std::unique_ptr<SPUContext> sctx, OpExecutor *executor,
                      SymbolScope *sscope, mlir::Operation *op,
                      SymbolTableEvent *event)
      : sctx_(std::move(sctx)),
        executor_(executor),
        sscope_(sscope),
        op_(op),
        event_(event) {
    // If a op has nested regions, it may depend on more values than operands
    // FIXME: (azheng) Implement a better notify mechanism
    if (op->getNumRegions() > 0) {
      const auto *current_region = op->getParentRegion();
      for (auto &r : op->getRegions()) {
        r.walk([&](mlir::Operation *nested_op) {
          for (const auto &o : nested_op->getOperands()) {
            if ((o.getDefiningOp() != nullptr) &&
                o.getDefiningOp()->getParentRegion() == current_region) {
              extra_dependencies_.emplace_back(o);
            }
          }
        });
      }
    }
  }

  bool ready() {
    return sscope_->hasValues(op_->getOperands()) &&
           sscope_->hasValues(extra_dependencies_);
  }

  void run() {
    // wait for this operation ready.
    if (op_->getNumOperands() > 0) {
      std::unique_lock lk(event_->mutex);
      event_->cv.wait(lk, [this] { return ready(); });
    }

    executor_->runKernel(sctx_.get(), sscope_, *op_);
    std::unique_lock lk(event_->mutex);
    event_->cv.notify_all();
  }
};

std::vector<spu::Value> runBlockParallel(OpExecutor *executor, SPUContext *sctx,
                                         SymbolScope *symbols,
                                         mlir::Block &block,
                                         absl::Span<spu::Value const> params,
                                         const ExecutionOptions &opts) {
  // The strategy is try to execute operations without dependency as much as
  // possible. For each (maybe) parallel operation we allocate a new hal
  // context.
  SymbolTableEvent st_event;
  std::vector<OpExecTask> tasks;
  std::vector<std::future<void>> futures;
  for (auto &op : block.without_terminator()) {
    tasks.emplace_back(sctx->fork(), executor, symbols, &op, &st_event);
  }

  futures.reserve(tasks.size());
  for (auto &task : tasks) {
    futures.emplace_back(
        std::async(std::launch::async, &OpExecTask::run, &task));
  }

  // Long story short....
  // std::future A throws
  // A.get() will propagate the exception to main thread.
  // main thread starts stack unwinding and tries to destroy of futures
  // std::vector<std::future>{...A...} all downstream of A is waiting on A...
  // thus hang during unwinding, so if a future throws, the only choice we have
  // is abort
  for (size_t idx = 0; idx < futures.size(); ++idx) {
    try {
      futures[idx].get();
    } catch (yacl::Exception &e) {
      SPDLOG_ERROR("OpExecTask {} run failed, reason = {}, stacktrace = {}",
                   idx, e.what(), e.stack_trace());
      spdlog::shutdown();  // Make sure log printed out....
      abort();
    } catch (...) {
      abort();  // WTF...jump the boat...
    }
  }

  if (auto *termOp = block.getTerminator()) {
    // TODO: enforce ReturnLike
    std::vector<spu::Value> results;
    results.reserve(termOp->getNumOperands());
    for (const auto operand : termOp->getOperands()) {
      results.emplace_back(symbols->lookupValue(operand));
    }
    return results;
  }

  // No terminator
  SPU_THROW("Should not be here");
}

}  // namespace spu::device
