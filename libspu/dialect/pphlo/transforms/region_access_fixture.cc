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

#include <memory>

#include "mlir/Pass/Pass.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {

namespace {

struct Deallocator {
 public:
  LogicalResult transformOp(Operation *op) {
    for (auto &r : op->getRegions()) {
      if (failed(transformRegion(r))) {
        return failure();
      }
    }

    const auto &operands = op->getOperands();

    if (op->getNumOperands() < 2 ||
        !op->hasTrait<::mlir::OpTrait::Elementwise>() ||
        std::all_of(operands.begin(), operands.end(), [](const auto &operand) {
          return operand.template getDefiningOp<ConstantOp>();
        })) {
      return success();
    }

    auto *op_region = op->getParentRegion();

    Value base_val;
    llvm::SmallVector<int64_t, 2> values_to_update;

    OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);

    for (const auto &[idx, operand] : llvm::enumerate(operands)) {
      // Get defining region
      auto *defining_op = operand.getDefiningOp();

      Region *defining_region = nullptr;

      if (defining_op != nullptr) {
        defining_region = defining_op->getParentRegion();
      }

      if (defining_op == nullptr || defining_region == op_region) {
        // BlockArg or op defined in current region can be a base val
        base_val = operand;
        continue;
      }

      if (defining_region != op_region) {
        // This op is accessing a variable out of op's region.
        // Insert a broadcast as to fix runtime shape mismatch during simd
        // region execution
        values_to_update.emplace_back(idx);
      }
    }

    if (!base_val) {
      return values_to_update.empty()
                 ? failure()   // same region however failed to pick base value
                 : success();  // can't pick base value since multi-level
                               // nesting
    }

    for (const auto &idx : values_to_update) {
      auto op_to_broadcast = op->getOperand(idx);
      auto b = builder.create<BroadcastShapeAsOp>(
          op->getLoc(), op_to_broadcast.getType(), op_to_broadcast, base_val);
      op->setOperand(idx, b);
    }

    return success();
  }

  LogicalResult transformBlock(Block &block) {
    for (auto &op : llvm::make_early_inc_range(block.without_terminator())) {
      auto opResult = transformOp(&op);
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

    // Transform function body.
    if (failed(transformRegion(op.getBody()))) {
      return failure();
    }

    return success();
  }
};

struct RegionAccessFixture
    : public RegionAccessFixtureBase<RegionAccessFixture> {
  void runOnOperation() override {
    if (failed(Deallocator().transformFuncOp(getOperation()))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRegionAccessFixture() {
  return std::make_unique<RegionAccessFixture>();
}

}  // namespace mlir::spu::pphlo
