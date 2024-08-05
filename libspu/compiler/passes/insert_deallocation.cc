// Copyright 2023 Ant Group Co., Ltd.
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

#include <memory>

#include "mlir/Analysis/Liveness.h"
#include "mlir/Pass/Pass.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/dialect/pphlo_ops.h"

#ifdef ENABLE_LIVENESS_DEBUG

#include "spdlog/spdlog.h"

void printLiveness(mlir::Liveness *liveness) {
  std::string buf;
  llvm::raw_string_ostream os(buf);

  liveness->print(os);

  SPDLOG_INFO("liveness = {}", os.str());
}

#endif

namespace mlir::pphlo {

namespace {

struct Deallocator {
private:
  std::unique_ptr<Liveness> top_liveness_;

public:
  LogicalResult transformOp(Operation *op,
                            const LivenessBlockInfo *block_liveness) {
    for (const auto &operand : op->getOperands()) {
      if (block_liveness->isLiveOut(operand) ||
          mlir::isa<BlockArgument>(operand)) {
        // skip live out values and block args
        continue;
      }

      if (operand.getDefiningOp()->getParentRegion() != op->getParentRegion()) {
        // This value is captured by current region, right now we do not handle
        // cross region ownership.. skip
        continue;
      }

      if (top_liveness_->isDeadAfter(operand, op)) {
        OpBuilder builder(op->getContext());
        builder.setInsertionPointAfter(op);
        builder.create<FreeOp>(op->getLoc(), operand);
      }
    }

    for (int64_t idx = 0; idx < op->getNumRegions(); ++idx) {
      if (failed(transformRegion(op->getRegion(idx)))) {
        return failure();
      }
    }

    return success();
  }

  LogicalResult transformBlock(Block &block) {
    const auto *block_liveness = top_liveness_->getLiveness(&block);
    for (auto &op : llvm::make_early_inc_range(block.without_terminator())) {
      auto opResult = transformOp(&op, block_liveness);
      if (failed(opResult)) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult transformRegion(Region &r) {
    for (auto &b : r.getBlocks()) {
      if (failed(transformBlock(b))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult transformFuncOp(func::FuncOp op) {
    if (op->getNumRegions() == 0) {
      return success();
    }

    top_liveness_ = std::make_unique<Liveness>(op);

    // Transform function body.
    if (failed(transformRegion(op.getBody()))) {
      return failure();
    }

    return success();
  }
};

struct InsertDeallocation : public InsertDeallocationBase<InsertDeallocation> {
  void runOnOperation() override {
    if (failed(Deallocator().transformFuncOp(getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInsertDeallocationOp() {
  return std::make_unique<InsertDeallocation>();
}

} // namespace mlir::pphlo
