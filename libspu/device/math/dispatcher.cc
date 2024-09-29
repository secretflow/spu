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

#include "libspu/device/math/dispatcher.h"

#include "absl/numeric/bits.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "libspu/core/trace.h"           // IWYU pragma: keep
#include "libspu/device/utils/utils.h"   // IWYU pragma: keep
#include "libspu/dialect/utils/utils.h"  // IWYU pragma: keep
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::device {

template <typename FN>
void applyUnaryFpFcn(PtType pt_type, const MemRef &in, MemRef &out, FN &&fn) {
  auto numel = in.numel();

  DISPATCH_FLOAT_PT_TYPES(pt_type, [&]() {
    MemRefView<ScalarT> _in(in);
    MemRefView<ScalarT> _out(out);
    for (int64_t idx = 0; idx < numel; ++idx) {
      _out[idx] = fn(_in[idx]);
    }
  });
}

template <typename FN>
void applyUnaryIntFcn(PtType pt_type, const MemRef &in, MemRef &out, FN &&fn) {
  auto numel = in.numel();

  DISPATCH_INT_PT_TYPES(pt_type, [&]() {
    MemRefView<ScalarT> _in(in);
    MemRefView<ScalarT> _out(out);
    for (int64_t idx = 0; idx < numel; ++idx) {
      _out[idx] = fn(_in[idx]);
    }
  });
}

template <typename FN>
void applyBinaryFpFcn(PtType pt_type, const MemRef &x, const MemRef &y,
                      MemRef &out, FN &&fn) {
  auto numel = x.numel();

  DISPATCH_FLOAT_PT_TYPES(pt_type, [&]() {
    MemRefView<ScalarT> _lhs(x);
    MemRefView<ScalarT> _rhs(y);
    MemRefView<ScalarT> _out(out);
    for (int64_t idx = 0; idx < numel; ++idx) {
      _out[idx] = fn(_lhs[idx], _rhs[idx]);
    }
  });
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::CosOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return std::cos(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::SinOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return std::sin(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::TanhOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return std::tanh(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::RsqrtOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return 1 / std::sqrt(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::RoundOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return std::round(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::FloorOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return std::floor(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::CtPopOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryIntFcn(
      getPtTypeFromMlirType(op.getType()), in, out, [](uint128_t x) {
        auto parts = yacl::DecomposeUInt128(x);
        return absl::popcount(parts.first) + absl::popcount(parts.second);
      });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::CeilOp &op, const ExecutionOptions &opts) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return std::ceil(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::CopySignOp &op, const ExecutionOptions &opts) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef out(lhs.eltype(), lhs.shape());

  applyBinaryFpFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, out,
                   [](double x, double y) {
                     if (y < 0) {
                       return -std::abs(x);
                     }
                     return std::abs(x);
                   });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::AbsFOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef out(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, out,
                  [](double x) { return std::abs(x); });

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::math::PowFOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef out(rhs.eltype(), rhs.shape());

  applyBinaryFpFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, out,
                   [](double x, double y) { return std::pow(x, y); });

  sscope->addValue(op.getResult(), std::move(out));
}

#include "libspu/device/utils/dispatch_template.cc.inc"

void dispatch(mlir::math::MathDialect *, SPUContext *sctx, SymbolScope *sscope,
              mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts) {
  dispatchOp<mlir::math::AbsFOp,      //
             mlir::math::CosOp,       //
             mlir::math::CopySignOp,  //
             mlir::math::CtPopOp,     //
             mlir::math::FloorOp,     //
             mlir::math::CeilOp,      //
             mlir::math::PowFOp,      //
             mlir::math::RsqrtOp,     //
             mlir::math::RoundOp,     //
             mlir::math::SinOp,       //
             mlir::math::TanhOp       //
             >(executor, sctx, sscope, op, opts);
}

}  // namespace spu::device