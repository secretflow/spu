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

#include "libspu/device/arith/dispatcher.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "libspu/core/trace.h"           // IWYU pragma: keep
#include "libspu/device/utils/utils.h"   // IWYU pragma: keep
#include "libspu/dialect/utils/utils.h"  // IWYU pragma: keep
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::device {

namespace {

template <typename FN>
void applyCastingFcn(PtType in_pt_type, PtType out_pt_type, const MemRef &in,
                     MemRef &out, FN &&fn) {
  auto numel = in.numel();

  DISPATCH_ALL_PT_TYPES(in_pt_type, [&]() {
    using in_scalar_t = ScalarT;
    MemRefView<in_scalar_t> _in(in);
    DISPATCH_ALL_PT_TYPES(out_pt_type, [&]() {
      MemRefView<ScalarT> _out(out);
      for (int64_t idx = 0; idx < numel; ++idx) {
        if constexpr (std::is_same_v<in_scalar_t, bool>) {
          _out[idx] = fn((std::uint8_t)_in[idx]);
        } else {
          _out[idx] = fn(_in[idx]);
        }
      }
    });
  });
}

template <typename FN>
void applyBinaryIntFcn(PtType pt_type, const MemRef &x, const MemRef &y,
                       MemRef &out, FN &&fn) {
  auto numel = x.numel();

  DISPATCH_SINT_PT_TYPES(pt_type, [&]() {
    MemRefView<ScalarT> _lhs(x);
    MemRefView<ScalarT> _rhs(y);
    MemRefView<ScalarT> _out(out);
    for (int64_t idx = 0; idx < numel; ++idx) {
      _out[idx] = fn(_lhs[idx], _rhs[idx]);
    }
  });
}

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

template <typename FN>
void applyCompareSIFcn(PtType pt_type, const MemRef &x, const MemRef &y,
                       MemRef &out, FN &&fn) {
  auto numel = x.numel();

  MemRefView<bool> _out(out);

  DISPATCH_SINT_PT_TYPES(pt_type, [&]() {
    MemRefView<ScalarT> _lhs(x);
    MemRefView<ScalarT> _rhs(y);
    for (int64_t idx = 0; idx < numel; ++idx) {
      _out[idx] = fn(_lhs[idx], _rhs[idx]);
    }
  });
}

template <typename FN>
void applyCompareFpFcn(PtType pt_type, const MemRef &x, const MemRef &y,
                       MemRef &out, FN &&fn) {
  auto numel = x.numel();

  MemRefView<bool> _out(out);

  DISPATCH_FLOAT_PT_TYPES(pt_type, [&]() {
    MemRefView<ScalarT> _lhs(x);
    MemRefView<ScalarT> _rhs(y);
    for (int64_t idx = 0; idx < numel; ++idx) {
      _out[idx] = fn(_lhs[idx], _rhs[idx]);
    }
  });
}

MemRef processTensorConstant(SPUContext *sctx,
                             const mlir::DenseElementsAttr &dea,
                             mlir::Type result_type) {
  const auto &type = dea.getType();
  const Shape &dst_shape = type.getShape();
  auto pt_type = getPtTypeFromMlirType(dea.getElementType());
  MemRef ret;
  // For 1-bit type, MLIR buffer is either 0 or 255
  // See
  // https://github.com/llvm/llvm-project/blob/3696941dae5cc5bb379c50eae6190e29f7edbbb1/mlir/include/mlir/IR/BuiltinAttributes.h#L188
  // We need to normalize the value to 0,1
  if (dea.getElementType().isInteger(1)) {
    if (dea.isSplat()) {
      ret = kernel::hal::_encode_int(sctx, dea.getSplatValue<bool>(), SE_1);
    } else {
      std::vector<uint8_t> buf(type.getNumElements());
      for (const auto &v : llvm::enumerate(dea.getValues<bool>())) {
        buf[v.index()] = static_cast<uint8_t>(v.value());
      }
      PtBufferView view(reinterpret_cast<const bool *>(buf.data()), pt_type,
                        dst_shape, makeCompactStrides(dst_shape));
      ret = kernel::hal::_encode_int(sctx, view, SE_1);
    }
  } else if (pt_type == PT_I128) {
    // APInt is not align to 16 bytes, which is required by int128_t, so make
    // a copy here.
    std::vector<int128_t> buffer(dea.isSplat() ? 1 : dea.getNumElements());
    std::memcpy(buffer.data(), dea.getRawData().data(), 16 * buffer.size());
    PtBufferView view(
        buffer.data(), pt_type, dea.isSplat() ? Shape() : dst_shape,
        dea.isSplat() ? Strides() : makeCompactStrides(dst_shape));
    auto se_type = getSemanticTypeFromMlirType(result_type);
    ret = kernel::hal::_encode_int(sctx, view, se_type);
  } else {
    auto numel = dea.isSplat() ? 1 : dea.getNumElements();
    std::vector<uint8_t> buffer(numel * SizeOf(pt_type));
    std::memcpy(buffer.data(), dea.getRawData().data(), buffer.size());
    PtBufferView view(
        buffer.data(), pt_type, dea.isSplat() ? Shape() : dst_shape,
        dea.isSplat() ? Strides() : makeCompactStrides(dst_shape));
    if (pt_type != PT_F16 && pt_type != PT_F32 && pt_type != PT_F64) {
      auto se_type = getSemanticTypeFromMlirType(result_type);
      ret = kernel::hal::_encode_int(sctx, view, se_type);
    } else {
      // For floating-point, directly encode as raw bytes, ToFxpOp will handle
      // encoding
      ret = kernel::hal::_copy_fp(sctx, view);
    }
  }
  if (dea.isSplat()) {
    ret = kernel::hal::broadcast_to(sctx, ret, dst_shape, {});
  }
  return ret;
}

}  // namespace

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::SelectOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getCondition());
  const auto &on_true = sscope->lookupValue(op.getTrueValue());
  const auto &on_false = sscope->lookupValue(op.getFalseValue());

  if (mlir::isa<mlir::IntegerType>(op.getCondition().getType())) {
    bool cond = kernel::hal::getScalarValue<bool>(sctx, in);

    MemRef ret = cond ? on_true : on_false;
    sscope->addValue(op.getResult(), std::move(ret));
  } else {
    // element-wise select
    // in case where select a private and a secret
    auto common_type =
        kernel::hal::_common_type(sctx, on_true.eltype(), on_false.eltype());
    MemRef ret(common_type, on_true.shape());
    auto on_true_ = kernel::hal::_cast_type(sctx, on_true, common_type);
    auto on_false_ = kernel::hal::_cast_type(sctx, on_false, common_type);
    auto cond = kernel::hal::dump_public_as_vec<uint8_t>(sctx, in);

    for (size_t idx = 0; idx < cond.size(); ++idx) {
      std::memcpy(
          &ret.at(idx),
          static_cast<bool>(cond[idx]) ? &on_true_.at(idx) : &on_false_.at(idx),
          ret.elsize());
    }
    sscope->addValue(op.getResult(), MemRef(ret));
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::ConstantOp &op, const ExecutionOptions &) {
  const auto &val = op.getValue();
  MemRef ret;
  if (auto tensor_type =
          mlir::dyn_cast<mlir::RankedTensorType>(val.getType())) {
    ret = processTensorConstant(sctx, mlir::cast<mlir::DenseElementsAttr>(val),
                                op->getResultTypes()[0]);
  } else {
    auto pt_type = getPtTypeFromMlirType(val.getType());

    // Scalar constant
    if (val.getType().isInteger(1)) {
      ret = kernel::hal::_encode_int(
          sctx, mlir::cast<mlir::BoolAttr>(val).getValue(), SE_1);
    } else if (pt_type == PT_I128) {
      // APInt is not align to 16 bytes, which is required by int128_t, so make
      // a copy here.
      int128_t buffer;

      std::memcpy(&buffer,
                  mlir::cast<mlir::IntegerAttr>(val).getValue().getRawData(),
                  16);

      PtBufferView view(&buffer, pt_type, Shape(), Strides());

      auto se_type = getSemanticTypeFromMlirType(op.getResult().getType());

      ret = kernel::hal::_encode_int(sctx, view, se_type);

    } else {
      std::vector<uint8_t> buffer(SizeOf(pt_type));
      if (pt_type != PT_F16 && pt_type != PT_F32 && pt_type != PT_F64) {
        std::memcpy(buffer.data(),
                    mlir::cast<mlir::IntegerAttr>(val).getValue().getRawData(),
                    buffer.size());
        PtBufferView view(buffer.data(), pt_type, Shape(), Strides());
        auto se_type = getSemanticTypeFromMlirType(op.getResult().getType());
        ret = kernel::hal::_encode_int(sctx, view, se_type);
      } else {
        // For floating-point, directly encode as raw bytes, ToFxpOp will handle
        // encoding
        std::memcpy(buffer.data(),
                    mlir::cast<mlir::FloatAttr>(val)
                        .getValue()
                        .bitcastToAPInt()
                        .getRawData(),
                    buffer.size());

        PtBufferView view(buffer.data(), pt_type, Shape(), Strides());
        ret = kernel::hal::_copy_fp(sctx, view);
      }
    }
  }

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::IndexCastOp &op, const ExecutionOptions &) {
  sscope->addValue(op.getResult(), sscope->lookupValue(op.getIn()));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::UIToFPOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getIn());

  auto in_pt_type = getPtTypeFromMlirType(op.getOperand().getType());
  auto out_pt_type = getPtTypeFromMlirType(op.getType());

  auto ret_setype = getSemanticTypeFromMlirType(op.getType());

  MemRef ret(makeType<mpc::Pub2kTy>(ret_setype), in.shape());

  applyCastingFcn(in_pt_type, out_pt_type, in, ret,
                  [](uint128_t in) { return static_cast<double>(in); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::SIToFPOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getIn());

  auto in_pt_type = getPtTypeFromMlirType(op.getOperand().getType());
  auto out_pt_type = getPtTypeFromMlirType(op.getType());

  auto ret_setype = getSemanticTypeFromMlirType(op.getType());

  MemRef ret(makeType<mpc::Pub2kTy>(ret_setype), in.shape());

  applyCastingFcn(in_pt_type, out_pt_type, in, ret,
                  [](int128_t in) { return static_cast<double>(in); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::FPToSIOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getIn());

  auto in_pt_type = getPtTypeFromMlirType(op.getOperand().getType());
  auto out_pt_type = getPtTypeFromMlirType(op.getType());

  auto ret_setype = getSemanticTypeFromMlirType(op.getType());

  MemRef ret(makeType<mpc::Pub2kTy>(ret_setype), in.shape());

  applyCastingFcn(in_pt_type, out_pt_type, in, ret,
                  [](double in) { return static_cast<int128_t>(in); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::AddIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](int128_t x, int128_t y) { return x + y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::AddFOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryFpFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                   [](double x, double y) { return x + y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::SubIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return x - y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::SubFOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryFpFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                   [](double x, double y) { return x - y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::AndIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return x & y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::MulIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return x * y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::MulFOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryFpFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                   [](double x, double y) { return x * y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::RemUIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return x % y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::RemFOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryFpFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                   [](double x, double y) { return std::remainder(x, y); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::ShLIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, int64_t y) { return x << y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::ShRUIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, int64_t y) { return x >> y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::ShRSIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](int128_t x, int64_t y) { return x >> y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::DivUIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return x / y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::DivFOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryFpFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                   [](double x, double y) { return x / y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::MinUIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return std::min(x, y); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::MinSIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](int128_t x, int128_t y) { return std::min(x, y); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::MaxUIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return std::max(x, y); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::MaxSIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](int128_t x, int128_t y) { return std::max(x, y); });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::OrIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return x | y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::XOrIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef ret(lhs.eltype(), lhs.shape());

  applyBinaryIntFcn(getPtTypeFromMlirType(op.getType()), lhs, rhs, ret,
                    [](uint128_t x, uint128_t y) { return x ^ y; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::NegFOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getOperand());

  MemRef ret(in.eltype(), in.shape());

  applyUnaryFpFcn(getPtTypeFromMlirType(op.getType()), in, ret,
                  [](double x) { return -x; });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::CmpFOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef out(makeType<mpc::Pub2kTy>(SE_1), lhs.shape());

  std::function<bool(double, double)> comp_fn;

  switch (op.getPredicate()) {
    case mlir::arith::CmpFPredicate::OEQ:
      comp_fn = [](double x, double y) { return x == y; };
      break;
    case mlir::arith::CmpFPredicate::OGE:
      comp_fn = [](double x, double y) { return x >= y; };
      break;
    case mlir::arith::CmpFPredicate::OGT:
      comp_fn = [](double x, double y) { return x > y; };
      break;
    case mlir::arith::CmpFPredicate::OLE:
      comp_fn = [](double x, double y) { return x <= y; };
      break;
    case mlir::arith::CmpFPredicate::OLT:
      comp_fn = [](double x, double y) { return x < y; };
      break;
    case mlir::arith::CmpFPredicate::ONE:
      comp_fn = [](double x, double y) { return x != y; };
      break;
    case mlir::arith::CmpFPredicate::UNO:
      comp_fn = [](double x, double y) {
        return std::isnan(x) && std::isnan(y);
      };
      break;
    default:
      SPU_THROW("Unknown CmpF direction {}",
                mlir::spu::mlirObjectToString(op.getPredicate()));
  }

  applyCompareFpFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs, out,
                    comp_fn);

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::CmpIOp &op, const ExecutionOptions &) {
  const auto &lhs = sscope->lookupValue(op.getLhs());
  const auto &rhs = sscope->lookupValue(op.getRhs());

  MemRef out(makeType<mpc::Pub2kTy>(SE_1), lhs.shape());

  switch (op.getPredicate()) {
    case mlir::arith::CmpIPredicate::slt: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) { return x < y; });
      break;
    }
    case mlir::arith::CmpIPredicate::sle: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) { return x <= y; });
      break;
    }
    case mlir::arith::CmpIPredicate::sgt: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) { return x > y; });
      break;
    }
    case mlir::arith::CmpIPredicate::sge: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) { return x >= y; });
      break;
    }
    case mlir::arith::CmpIPredicate::eq: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) { return x == y; });
      break;
    }
    case mlir::arith::CmpIPredicate::ne: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) { return x != y; });
      break;
    }
    case mlir::arith::CmpIPredicate::ult: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) {
                          return static_cast<uint128_t>(x) <
                                 static_cast<uint128_t>(y);
                        });
      break;
    }
    case mlir::arith::CmpIPredicate::ule: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) {
                          return static_cast<uint128_t>(x) <=
                                 static_cast<uint128_t>(y);
                        });
      break;
    }
    case mlir::arith::CmpIPredicate::ugt: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) {
                          return static_cast<uint128_t>(x) >
                                 static_cast<uint128_t>(y);
                        });
      break;
    }
    case mlir::arith::CmpIPredicate::uge: {
      applyCompareSIFcn(getPtTypeFromMlirType(op.getLhs().getType()), lhs, rhs,
                        out, [](int128_t x, int128_t y) {
                          return static_cast<uint128_t>(x) >=
                                 static_cast<uint128_t>(y);
                        });
      break;
    }
    default:
      SPU_THROW("Unknown CmpF direction {}",
                mlir::spu::mlirObjectToString(op.getPredicate()));
  }

  sscope->addValue(op.getResult(), std::move(out));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::ExtUIOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getIn());

  auto ret_setype = getSemanticTypeFromMlirType(op.getType());
  auto numel = in.numel();

  MemRef ret(makeType<mpc::Pub2kTy>(ret_setype), in.shape());

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using in_scalar_t = std::make_unsigned_t<ScalarT>;
    MemRefView<in_scalar_t> _in(in);
    DISPATCH_ALL_STORAGE_TYPES(ret.eltype().storage_type(), [&]() {
      using out_scalar_t = std::make_unsigned_t<ScalarT>;
      MemRefView<out_scalar_t> _out(ret);
      for (int64_t idx = 0; idx < numel; ++idx) {
        _out[idx] = _in[idx];
      }
    });
  });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::ExtSIOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getIn());

  auto ret_setype = getSemanticTypeFromMlirType(op.getType());
  auto numel = in.numel();

  MemRef ret(makeType<mpc::Pub2kTy>(ret_setype), in.shape());

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using in_scalar_t = std::make_signed_t<ScalarT>;
    MemRefView<in_scalar_t> _in(in);
    DISPATCH_ALL_STORAGE_TYPES(ret.eltype().storage_type(), [&]() {
      using out_scalar_t = std::make_signed_t<ScalarT>;
      MemRefView<out_scalar_t> _out(ret);
      for (int64_t idx = 0; idx < numel; ++idx) {
        _out[idx] = _in[idx];
      }
    });
  });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::arith::TruncIOp &op, const ExecutionOptions &) {
  const auto &in = sscope->lookupValue(op.getIn());

  auto ret_setype = getSemanticTypeFromMlirType(op.getType());
  auto numel = in.numel();

  MemRef ret(makeType<mpc::Pub2kTy>(ret_setype), in.shape());

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _in(in);
    DISPATCH_ALL_STORAGE_TYPES(ret.eltype().storage_type(), [&]() {
      MemRefView<ScalarT> _out(ret);
      for (int64_t idx = 0; idx < numel; ++idx) {
        _out[idx] = _in[idx];
      }
    });
  });

  sscope->addValue(op.getResult(), std::move(ret));
}

#include "libspu/device/utils/dispatch_template.cc.inc"

void dispatch(mlir::arith::ArithDialect *, SPUContext *sctx,
              SymbolScope *sscope, mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts) {
  dispatchOp<mlir::arith::AddIOp,       //
             mlir::arith::AddFOp,       //
             mlir::arith::AndIOp,       //
             mlir::arith::CmpFOp,       //
             mlir::arith::CmpIOp,       //
             mlir::arith::DivUIOp,      //
             mlir::arith::DivFOp,       //
             mlir::arith::ExtUIOp,      //
             mlir::arith::ExtSIOp,      //
             mlir::arith::SelectOp,     //
             mlir::arith::ConstantOp,   //
             mlir::arith::IndexCastOp,  //
             mlir::arith::MaxSIOp,      //
             mlir::arith::MaxUIOp,      //
             mlir::arith::MinSIOp,      //
             mlir::arith::MinUIOp,      //
             mlir::arith::MulIOp,       //
             mlir::arith::MulFOp,       //
             mlir::arith::NegFOp,       //
             mlir::arith::OrIOp,        //
             mlir::arith::RemUIOp,      //
             mlir::arith::RemFOp,       //
             mlir::arith::ShLIOp,       //
             mlir::arith::ShRUIOp,      //
             mlir::arith::ShRSIOp,      //
             mlir::arith::SubIOp,       //
             mlir::arith::SubFOp,       //
             mlir::arith::UIToFPOp,     //
             mlir::arith::SIToFPOp,     //
             mlir::arith::FPToSIOp,     //
             mlir::arith::XOrIOp,       //
             mlir::arith::TruncIOp      //
             >(executor, sctx, sscope, op, opts);
}

}  // namespace spu::device