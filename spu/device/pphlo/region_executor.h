// Copyright 2022 Ant Group Co., Ltd.
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
#include <optional>

#include "spu/device/frame.h"
#include "spu/device/pphlo/type_checker.h"
#include "spu/device/pphlo/xla_verifier.h"
#include "spu/device/profiler.h"
#include "spu/dialect/pphlo_ops.h"
#include "spu/hal/constants.h"
#include "spu/hal/context.h"
#include "spu/hal/type_cast.h"

namespace spu::device::pphlo {

class RegionExecutor {
public:
  explicit RegionExecutor(HalContext *ctx, Frame *frame,
                          std::shared_ptr<Profiler> profiler)
      : hctx_(ctx), frame_(frame), profiler_(std::move(profiler)),
        type_checker_(std::make_shared<PPHloTypeChecker>()) {
    frame->enterRegion();

    if (ctx->feature_control().enable_xla_verifier) {
      verifier_ = std::make_unique<XlaVerifier>(ctx);
    }
  }

  ~RegionExecutor() { frame_->leaveRegion(); }

  std::vector<hal::Value> executeRegion(mlir::Region &region,
                                        absl::Span<const hal::Value> inputs);

  HalContext *getContext() const { return hctx_; }

private:
  std::vector<hal::Value> executeBlock(mlir::Block &block);
  std::vector<hal::Value> executeTerminator(mlir::Operation &op);

  void debug_print(mlir::Operation &op);

  template <typename OpT, typename... MoreOpT>
  void dispatchOp(mlir::Operation &op) {
    if (auto casted = llvm::dyn_cast<OpT>(op)) {
      if (!suppress_pphlo_trace_ &&
          (hctx_->rt_config().enable_pphlo_trace() ||
           hctx_->feature_control().enable_xla_verifier)) {
        debug_print(op);
      }

      std::optional<Timer> tp;
      if (hctx_->rt_config().enable_pphlo_profile()) {
        tp = profiler_->start();
      }

      // Execute op
      execute(casted);

      if (tp.has_value()) {
        profiler_->end(op.getName().getStringRef(), *tp);
      }

      if (verifier_) {
        // Collect inputs
        std::vector<hal::Value> ins;
        for (auto operand : op.getOperands()) {
          ins.emplace_back(lookupValue(operand));
        }
        std::vector<hal::Value> outs;
        for (auto operand : op.getResults()) {
          outs.emplace_back(lookupValue(operand));
        }

        verifier_->verify(casted, ins, outs);
      }
    } else {
      if constexpr (!sizeof...(MoreOpT)) {
        // If there is no more op types to dispatch, and the previous cast
        // fails..print error message
        errorUnknownOp(op);
      } else {
        dispatchOp<MoreOpT...>(op);
      }
    }
  }

  /// Unary ops
  void execute(mlir::pphlo::ReciprocalOp &op);
  void execute(mlir::pphlo::NegOp &op);
  void execute(mlir::pphlo::ExpOp &op);
  void execute(mlir::pphlo::LogOp &op);
  void execute(mlir::pphlo::Log1pOp &op);
  void execute(mlir::pphlo::CeilOp &op);
  void execute(mlir::pphlo::FloorOp &op);
  void execute(mlir::pphlo::AbsOp &op);
  void execute(mlir::pphlo::TransposeOp &op);
  void execute(mlir::pphlo::LogisticOp &op);
  void execute(mlir::pphlo::NotOp &op);
  void execute(mlir::pphlo::TanhOp &op);
  void execute(mlir::pphlo::RsqrtOp &op);

  /// Binary ops
  void execute(mlir::pphlo::EqualOp &op);
  void execute(mlir::pphlo::LessOp &op);
  void execute(mlir::pphlo::GreaterOp &op);

  void execute(mlir::pphlo::AddOp &op);
  void execute(mlir::pphlo::SubOp &op);
  void execute(mlir::pphlo::MulOp &op);
  void execute(mlir::pphlo::MixedMulOp &op);
  void execute(mlir::pphlo::PowOp &op);
  void execute(mlir::pphlo::RemOp &op);
  void execute(mlir::pphlo::MaxOp &op);
  void execute(mlir::pphlo::MinOp &op);
  void execute(mlir::pphlo::DotOp &op);
  void execute(mlir::pphlo::MixedDotOp &op);
  void execute(mlir::pphlo::ShiftLeftOp &op);
  void execute(mlir::pphlo::ShiftRightArithmeticOp &op);
  void execute(mlir::pphlo::ShiftRightLogicalOp &op);

  /// Ternary ops
  void execute(mlir::pphlo::ClampOp &op);

  /// Logical ops
  void execute(mlir::pphlo::AndOp &op);
  void execute(mlir::pphlo::OrOp &op);
  void execute(mlir::pphlo::XorOp &op);

  /// Shape ops
  void execute(mlir::pphlo::BroadcastOp &op);
  void execute(mlir::pphlo::ReshapeOp &op);
  void execute(mlir::pphlo::ConcatenateOp &op);
  void execute(mlir::pphlo::SliceOp &op);
  void execute(mlir::pphlo::GatherOp &op);
  void execute(mlir::pphlo::PadOp &op);
  void execute(mlir::pphlo::ReverseOp &op);

  /// Data generator ops
  void execute(mlir::pphlo::ConstOp &op);
  void execute(mlir::pphlo::IotaOp &op);

  /// Other ops
  void execute(mlir::pphlo::RngUniformOp &op);
  void execute(mlir::pphlo::ConvertOp &op);
  void execute(mlir::pphlo::BitcastConvertOp &op);
  void execute(mlir::pphlo::ConvOp &op);
  void execute(mlir::pphlo::SortOp &op);
  void execute(mlir::pphlo::DynamicUpdateSliceOp &op);
  void execute(mlir::pphlo::DynamicSliceOp &op);

  /// Reduce ops
  void execute(mlir::pphlo::ReduceOp &op);
  void execute(mlir::pphlo::ReduceWindowOp &op);

  /// Control flow ops
  void execute(mlir::pphlo::WhileOp &op);
  void execute(mlir::pphlo::IfOp &op);

  /// Debug ops
  void execute(mlir::pphlo::DbgPrintOp &op);

  /// Lowered ops (All these ops will throw at run time)
  void execute(mlir::pphlo::SqrtOp &op);
  void execute(mlir::pphlo::SelectOp &op);
  void execute(mlir::pphlo::SelectAndScatterOp &op);
  void execute(mlir::pphlo::ReturnOp &op);
  void execute(mlir::pphlo::NotEqualOp &op);
  void execute(mlir::pphlo::LessEqualOp &op);
  void execute(mlir::pphlo::GreaterEqualOp &op);
  void execute(mlir::pphlo::DivOp &op);
  void errorUnknownOp(mlir::Operation &op);

  Frame *getFrame() { return frame_; }

  const hal::Value &lookupValue(::mlir::Value v) const;

  HalContext *hctx_{nullptr};
  Frame *frame_{nullptr};
  std::shared_ptr<Profiler> profiler_;
  mlir::pphlo::TypeTools type_tools_;
  std::shared_ptr<PPHloTypeChecker> type_checker_;
  std::unique_ptr<XlaVerifier> verifier_;

  //
  bool suppress_type_check_ = false;
  bool suppress_pphlo_trace_ = false;
};

} // namespace spu::device::pphlo
