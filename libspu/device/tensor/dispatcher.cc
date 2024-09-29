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

#include "libspu/device/tensor/dispatcher.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/core/trace.h"           // IWYU pragma: keep
#include "libspu/device/utils/utils.h"   // IWYU pragma: keep
#include "libspu/dialect/utils/utils.h"  // IWYU pragma: keep
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::device {

void execute(OpExecutor *, SPUContext *, SymbolScope *sscope,
             mlir::tensor::EmptyOp &op, const ExecutionOptions &) {
  // Empty op is just a dummy op to satisfy linalg ops now. So do nothing, no op
  // should lookup values from empty at this point.
  sscope->addValue(op.getResult(), MemRef());
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::ReshapeOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getSource());

  auto to_shape =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType())
          .getShape();

  sscope->addValue(op.getResult(), kernel::hal::reshape(sctx, in, to_shape));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::ConcatOp &op, const ExecutionOptions &) {
  std::vector<MemRef> ins;

  for (int64_t idx = 0; idx < op.getNumOperands(); ++idx) {
    ins.emplace_back(sscope->lookupValue(op.getOperand(idx)));
  }

  auto ret = kernel::hal::concatenate(sctx, ins, op.getDim());

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::ExtractSliceOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getSource());

  Index offsets(op.getStaticOffsets());

  for (size_t idx = 0, d_idx = 0; idx < offsets.size(); ++idx) {
    if (offsets[idx] == mlir::ShapedType::kDynamic) {
      offsets[idx] = kernel::hal::getScalarValue<int64_t>(
          sctx, sscope->lookupValue(op.getOffsets()[d_idx++]));
    }
  }

  auto ret = kernel::hal::slice(sctx, in, offsets, op.getStaticSizes(),
                                op.getStaticStrides());

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::ExtractOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getTensor());

  if (auto linalg_reduce = mlir::dyn_cast<mlir::linalg::ReduceOp>(
          op->getParentRegion()->getParentOp())) {
    if (llvm::is_contained(
            linalg_reduce.getCombiner().back().getTerminator()->getOperands(),
            op.getResult())) {
      // FIXME: this is a hack to handle extract in reduce body
      sscope->addValue(op.getResult(), in);
      return;
    }
  }
  MemRef ret;

  const auto &indices = op.getIndices();
  if (indices.empty()) {
    ret = kernel::hal::reshape(sctx, in, {});
  } else {
    SPU_THROW("Unimpl op = {}", mlir::spu::mlirObjectToString(op));
  }
  sscope->addValue(op.getResult(), in);
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::FromElementsOp &op, const ExecutionOptions &) {
  if (op.getElements().size() == 1) {
    sscope->addValue(op.getResult(), sscope->lookupValue(op.getElements()[0]));
    return;
  }
  SPU_UNIMPL;
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::CollapseShapeOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getSrc());

  auto ret = kernel::hal::reshape(
      sctx, in, mlir::cast<mlir::ShapedType>(op.getResultType()).getShape());

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::InsertSliceOp &op, const ExecutionOptions &) {
  const auto strides = op.getStaticStrides();

  Index offsets(op.getStaticOffsets());

  for (size_t idx = 0, d_idx = 0; idx < offsets.size(); ++idx) {
    if (offsets[idx] == mlir::ShapedType::kDynamic) {
      offsets[idx] = kernel::hal::getScalarValue<int64_t>(
          sctx, sscope->lookupValue(op.getOffsets()[d_idx++]));
    }
  }

  auto ret = kernel::hal::insert_slice(sctx, sscope->lookupValue(op.getDest()),
                                       sscope->lookupValue(op.getSource()),
                                       offsets, strides, false);

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::PadOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getSource());

  auto yield = mlir::dyn_cast<mlir::tensor::YieldOp>(
      op.getRegion().getBlocks().back().back());

  SPU_ENFORCE(yield);

  const auto &pad_v = sscope->lookupValue(yield.getValue());

  auto ret =
      kernel::hal::pad(sctx, in, pad_v, op.getStaticLow(), op.getStaticHigh());

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::tensor::BitcastOp &op, const ExecutionOptions &) {
  auto in = sscope->lookupValue(op.getOperand());

  auto out = in;

  out.eltype().as<BaseRingType>()->set_semantic_type(
      getSemanticTypeFromMlirType(op.getType()));

  sscope->addValue(op.getResult(), std::move(out));
}

#include "libspu/device/utils/dispatch_template.cc.inc"

void dispatch(mlir::tensor::TensorDialect *, SPUContext *sctx,
              SymbolScope *sscope, mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts) {
  dispatchOp<mlir::tensor::BitcastOp,        //
             mlir::tensor::CollapseShapeOp,  //
             mlir::tensor::EmptyOp,          //
             mlir::tensor::ExtractSliceOp,   //
             mlir::tensor::ExtractOp,        //
             mlir::tensor::FromElementsOp,   //
             mlir::tensor::ConcatOp,         //
             mlir::tensor::PadOp,            //
             mlir::tensor::ReshapeOp,        //
             mlir::tensor::InsertSliceOp     //
             >(executor, sctx, sscope, op, opts);
}

}  // namespace spu::device