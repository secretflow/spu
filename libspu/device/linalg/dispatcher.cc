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

#include "libspu/device/linalg/dispatcher.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/RegionUtils.h"

#include "libspu/core/trace.h"           // IWYU pragma: keep
#include "libspu/device/utils/utils.h"   // IWYU pragma: keep
#include "libspu/dialect/utils/utils.h"  // IWYU pragma: keep
#include "libspu/kernel/hal/reducer.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/mpc/utils/linalg.h"

namespace spu::device {

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::linalg::ReduceOp &op, const ExecutionOptions &opts) {
  int64_t num_args = op->getNumOperands() / 2;
  Axes dimensions_to_reduce = op.getDimensions();

  std::vector<spu::MemRef> input_args(num_args);
  std::vector<spu::MemRef> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = sscope->lookupValue(op.getInputs()[i]);
    init_values[i] = sscope->lookupValue(op.getInits()[i]);
  }

  bool canIgnoreInitialValue =
      std::none_of(dimensions_to_reduce.begin(), dimensions_to_reduce.end(),
                   [](int64_t d) { return d == 0; });

  llvm::SetVector<mlir::Value> values;
  mlir::getUsedValuesDefinedAbove(op.getCombiner(), values);

  SymbolScope bcast_scope(sscope);

  std::vector<spu::MemRef> ret = kernel::hal::Reduce(
      sctx, input_args, init_values, dimensions_to_reduce,
      [&](absl::Span<const spu::MemRef> lhs,
          absl::Span<const spu::MemRef> rhs) {
        std::vector<spu::MemRef> operands;
        operands.reserve(lhs.size() + rhs.size());
        operands.insert(operands.end(), lhs.begin(), lhs.end());
        operands.insert(operands.end(), rhs.begin(), rhs.end());
        return runRegion(executor, sctx, &bcast_scope, op.getCombiner(),
                         operands);
      },
      [&](const Shape &bcast_shape) {
        for (auto v : values) {
          auto scalar = sscope->lookupValue(v);
          SPU_ENFORCE(scalar.numel() == 1);
          bcast_scope.addValue(
              v, kernel::hal::broadcast_to(sctx, scalar, bcast_shape));
        }
      },
      canIgnoreInitialValue);

  const auto &output_shape =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0])
          .getShape();

  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    sscope->addValue(op->getResult(idx),
                     kernel::hal::reshape(sctx, ret[idx], output_shape));
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::linalg::BroadcastOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getInput());
  auto to_type =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult()[0].getType());

  auto ret = kernel::hal::broadcast_to(sctx, in, to_type.getShape(),
                                       op.getDimensions());

  sscope->addValue(op.getResult()[0], std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::linalg::TransposeOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getInput());

  auto ret = kernel::hal::transpose(sctx, in, op.getPermutation());

  sscope->addValue(op->getResult(0), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::linalg::FillOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getInputs()[0]);

  auto to_type =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResults()[0].getType());

  auto ret = kernel::hal::expand(sctx, in, to_type.getShape());

  sscope->addValue(op.getResults()[0], std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::linalg::MatmulOp &op, const ExecutionOptions &opts) {
  const auto &lhs = sscope->lookupValue(op.getInputs()[0]);
  const auto &rhs = sscope->lookupValue(op.getInputs()[1]);

  bool is_float =
      mlir::isa<mlir::FloatType>(mlir::getElementTypeOrSelf(op.getType(0)));

  const auto M = lhs.shape()[0];
  const auto K = lhs.shape()[1];
  const auto N = rhs.shape()[1];

  MemRef ret(lhs.eltype(), {M, N});

  DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
    const auto lhs_stride_scale = lhs.elsize() / sizeof(ScalarT);
    const auto rhs_stride_scale = rhs.elsize() / sizeof(ScalarT);
    const auto ret_stride_scale = ret.elsize() / sizeof(ScalarT);

    const auto LDA = lhs_stride_scale * lhs.strides()[0];
    const auto IDA = lhs_stride_scale * lhs.strides()[1];
    const auto LDB = rhs_stride_scale * rhs.strides()[0];
    const auto IDB = rhs_stride_scale * rhs.strides()[1];
    const auto LDC = ret_stride_scale * ret.strides()[0];
    const auto IDC = ret_stride_scale * ret.strides()[1];

    if (is_float) {
      if (sizeof(ScalarT) == 4) {
        spu::mpc::linalg::matmul(M, N, K, lhs.data<const float>(), LDA, IDA,
                                 rhs.data<const float>(), LDB, IDB,
                                 ret.data<float>(), LDC, IDC);
      } else if (sizeof(ScalarT) == 16) {
        spu::mpc::linalg::matmul(M, N, K, lhs.data<const double>(), LDA, IDA,
                                 rhs.data<const double>(), LDB, IDB,
                                 ret.data<double>(), LDC, IDC);
      } else {
        SPU_THROW("Unhandled float type = {}",
                  mlir::spu::mlirObjectToString(op.getType(0)));
      }
    } else {
      spu::mpc::linalg::matmul(M, N, K, lhs.data<const ScalarT>(), LDA, IDA,
                               rhs.data<const ScalarT>(), LDB, IDB,
                               ret.data<ScalarT>(), LDC, IDC);
    }
  });

  sscope->addValue(op.getResults()[0], ret);
}

int64_t evalAffineExpr(mlir::AffineExpr expr, absl::Span<const int64_t> ivs) {
  return llvm::TypeSwitch<mlir::AffineExpr, int64_t>(expr)
      .Case([&](mlir::AffineDimExpr e) { return ivs[e.getPosition()]; })
      .Case<mlir::AffineConstantExpr>(
          [](mlir::AffineConstantExpr e) { return e.getValue(); })
      .Case<mlir::AffineBinaryOpExpr>([&](mlir::AffineBinaryOpExpr e) {
        auto lhs = evalAffineExpr(e.getLHS(), ivs);
        auto rhs = evalAffineExpr(e.getRHS(), ivs);

        switch (e.getKind()) {
          case mlir::AffineExprKind::Add:
            return lhs + rhs;
          case mlir::AffineExprKind::Mul:
            return lhs * rhs;
          case mlir::AffineExprKind::Mod:
            return lhs % rhs;
          case mlir::AffineExprKind::FloorDiv:
            return lhs / rhs;
          case mlir::AffineExprKind::CeilDiv:
            return static_cast<int64_t>(
                std::ceil(static_cast<float>(lhs) / static_cast<float>(rhs)));
          default:
            SPU_UNIMPL;
        }
      });
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::linalg::GenericOp &op, const ExecutionOptions &opts) {
  auto operands = llvm::map_to_vector(op.getInputs(), [&](mlir::Value operand) {
    return sscope->lookupValue(operand);
  });

  auto results = llvm::map_to_vector(op.getOutputs(), [&](mlir::Value output) {
    auto v = sscope->lookupValue(output);
    return v.isUninitialized() ? v : v.clone();
  });

  auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(op.getOperation());

  auto range = linalgOp.getStaticLoopRanges();

  size_t loop_depth = range.size();

  auto computeAffineIndex = [&](mlir::OpOperand &operand,
                                absl::Span<const int64_t> ivs) {
    auto affine_map = linalgOp.getMatchingIndexingMap(&operand);
    auto rank =
        mlir::cast<mlir::RankedTensorType>(operand.get().getType()).getRank();

    Index idx(rank);
    for (int64_t r = 0; r < rank; ++r) {
      mlir::AffineExpr expr = affine_map.getResult(r);
      idx[r] = evalAffineExpr(expr, ivs);
    }
    return idx;
  };

  auto getScalar = [&](mlir::OpOperand &operand, MemRef &value,
                       absl::Span<const int64_t> ivs) {
    if (value.isUninitialized()) {
      return value;
    }
    Index idx = computeAffineIndex(operand, ivs);
    auto o = value.slice(idx, Shape(idx.size(), 1), {});
    return o.reshape({});
  };

  auto getScalarOperand = [&](size_t operand_index,
                              absl::Span<const int64_t> ivs) {
    auto &operand = op->getOpOperand(operand_index);
    auto value = operands[operand_index];
    return getScalar(operand, value, ivs);
  };

  auto getScalarResult = [&](size_t result_index,
                             absl::Span<const int64_t> ivs) {
    auto value = results[result_index];
    if (value.isUninitialized()) {
      return value;
    }

    auto &operand = op->getOpOperand(result_index + op.getInputs().size());
    Index idx = computeAffineIndex(operand, ivs);
    auto o = value.slice(idx, Shape(idx.size(), 1), {});
    return o.reshape({});
  };

  auto updateResultWithScalar = [&](size_t result_index,
                                    absl::Span<const int64_t> ivs,
                                    const MemRef &update) {
    auto &operand = op->getOpOperand(result_index + op.getInputs().size());
    auto value = results[result_index];

    if (value.isUninitialized()) {
      results[result_index] = update.expand(
          mlir::cast<mlir::ShapedType>(operand.get().getType()).getShape());
      return;
    }

    Index idx = computeAffineIndex(operand, ivs);

    results[result_index] = kernel::hal::insert_slice(
        sctx, value, update, idx, Strides(idx.size(), 1), false);
  };

  std::vector<int64_t> iv(loop_depth, 0);
  std::function<void(size_t)> looper;

  looper = [&](size_t depth) {
    if (depth == loop_depth) {
      auto &block_to_run = op.getRegion().front();
      SymbolScope new_scope(sscope);
      for (size_t idx = 0; idx < op.getInputs().size(); ++idx) {
        new_scope.addValue(block_to_run.getArgument(idx),
                           getScalarOperand(idx, iv));
      }
      for (size_t idx = 0; idx < op.getOutputs().size(); ++idx) {
        new_scope.addValue(
            block_to_run.getArgument(idx + op.getInputs().size()),
            getScalarResult(idx, iv));
      }
      auto results =
          runBlock(executor, sctx, &new_scope, block_to_run, {}, opts);

      for (size_t idx = 0; idx < op.getOutputs().size(); ++idx) {
        updateResultWithScalar(idx, iv, results[idx]);
      }
      return;
    }
    for (iv[depth] = 0; iv[depth] < range[depth]; ++iv[depth]) {
      looper(depth + 1);
    }
  };

  looper(0);

  for (auto [idx, value] : llvm::enumerate(op->getResults())) {
    sscope->addValue(value, results[idx]);
  }
}

#include "libspu/device/utils/dispatch_template.cc.inc"

void dispatch(mlir::linalg::LinalgDialect *, SPUContext *sctx,
              SymbolScope *sscope, mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts) {
  dispatchOp<mlir::linalg::ReduceOp,     //
             mlir::linalg::BroadcastOp,  //
             mlir::linalg::FillOp,       //
             mlir::linalg::MatmulOp,     //
             mlir::linalg::TransposeOp,  //
             mlir::linalg::GenericOp     //
             >(executor, sctx, sscope, op, opts);
}

}  // namespace spu::device