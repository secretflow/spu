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

#include "libspu/device/ring/dispatcher.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "libspu/core/trace.h"          // IWYU pragma: keep
#include "libspu/device/utils/utils.h"  // IWYU pragma: keep
#include "libspu/dialect/ring/IR/ops.h"
#include "libspu/dialect/utils/utils.h"  // IWYU pragma: keep
#include "libspu/kernel/hal/indexing.h"
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/reducer.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::device {

namespace ring {

void do_type_checker(mlir::Value key, const spu::MemRef &val,
                     const ExecutionOptions &opts) {
  if (val.buf() == nullptr) {
    return;
  }
  if (opts.do_type_check) {
    SPU_ENFORCE(val.eltype().semantic_type() != SE_INVALID);
    SPU_ENFORCE(val.eltype().storage_type() != ST_INVALID);
    SPU_ENFORCE(val.vtype() != VIS_INVALID, "val storage type = {}",
                val.eltype());

    const auto mlir_type = key.getType();
    bool is_secret = mlir::spu::ring::isSecret(mlir_type);
    SPU_ENFORCE(
        (!is_secret && val.isPublic()) || (is_secret && !val.isPublic()),
        "Runtime visiblity mismatch, expected={}, got={}", is_secret,
        val.vtype());
    {
      const auto &mlir_shape =
          mlir::dyn_cast<mlir::RankedTensorType>(mlir_type).getShape();
      const auto &spu_shape = val.shape();

      SPU_ENFORCE(mlir_shape.size() == spu_shape.size(),
                  "Runtime shape mismatch, expected={}, got={}",
                  fmt::join(mlir_shape, "x"), fmt::join(spu_shape, "x"));

      for (size_t idx = 0; idx < mlir_shape.size(); ++idx) {
        SPU_ENFORCE(mlir_shape[idx] == spu_shape[idx],
                    "Runtime shape mismatch at dim {}, expected={}, got={}",
                    idx, fmt::join(mlir_shape, "x"), fmt::join(spu_shape, "x"));
      }
    }
  }
}

spu::MemRef lookupValue(SymbolScope *scope, mlir::Value key,
                        const ExecutionOptions &opts) {
  auto val = scope->lookupValue(key);
  do_type_checker(key, val, opts);
  return val;
}

void addValue(SymbolScope *scope, mlir::Value key, const spu::MemRef &val,
              const ExecutionOptions &opts) {
  do_type_checker(key, val, opts);
  scope->addValue(key, val);
}

void addValue(SymbolScope *scope, mlir::Value key, spu::MemRef &&val,
              const ExecutionOptions &opts) {
  do_type_checker(key, val, opts);
  scope->addValue(key, val);
}

#define STANDARD_UNARY_OP_EXEC_IMPL(OpName, Name)                           \
  void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,         \
               mlir::spu::ring::OpName &op, const ExecutionOptions &opts) { \
    const auto &in = lookupValue(sscope, op.getOperand(), opts);            \
    MemRef ret;                                                             \
    if (in.isSecret()) {                                                    \
      ret = Name##_s(sctx, in);                                             \
    } else if (in.isPrivate()) {                                            \
      ret = Name##_v(sctx, in);                                             \
    } else {                                                                \
      SPU_THROW("Should not reach here, in type {}", in.eltype());          \
    }                                                                       \
    addValue(sscope, op.getResult(), std::move(ret), opts);                 \
  }

#define OPTIONAL_UNARY_KERNEL(OpName, KernelName)                              \
  void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,            \
               mlir::spu::ring::OpName &op, const ExecutionOptions &opts) {    \
    const auto &in = lookupValue(sscope, op.getOperand(), opts);               \
    auto z = KernelName(sctx, in);                                             \
    if (!z.has_value()) {                                                      \
      SPU_THROW("Illegal instruction: {}", mlir::spu::mlirObjectToString(op)); \
    }                                                                          \
    addValue(sscope, op.getResult(), *z, opts);                                \
  }

STANDARD_UNARY_OP_EXEC_IMPL(MsbOp, kernel::hal::_msb)
STANDARD_UNARY_OP_EXEC_IMPL(NotOp, kernel::hal::_not)
STANDARD_UNARY_OP_EXEC_IMPL(NegOp, kernel::hal::_negate)
#undef STANDARD_UNARY_OP_EXEC_IMPL

#define STANDARD_BINARY_COMMUTATIVE_OP_EXEC_IMPL(OpName, Name)                 \
  void execute(OpExecutor *, SPUContext *ctx, SymbolScope *sscope,             \
               mlir::spu::ring::OpName &op, const ExecutionOptions &opts) {    \
    const auto &x = lookupValue(sscope, op.getLhs(), opts);                    \
    const auto &y = lookupValue(sscope, op.getRhs(), opts);                    \
    MemRef ret;                                                                \
    if (x.isPrivate() && y.isPrivate()) { /*VV*/                               \
      ret = Name##_vv(ctx, x, y);                                              \
    } else if (x.isSecret() && y.isSecret()) { /*SS*/                          \
      ret = Name##_ss(ctx, x, y);                                              \
    } else if (x.isSecret() && y.isPublic()) { /*SP*/                          \
      ret = Name##_sp(ctx, x, y);                                              \
    } else if (x.isPublic() && y.isSecret()) { /*PS*/                          \
      /* commutative, swap args */                                             \
      ret = Name##_sp(ctx, y, x);                                              \
    } else if (x.isPrivate() && y.isPublic()) { /*VP*/                         \
      ret = Name##_vp(ctx, x, y);                                              \
    } else if (x.isPublic() && y.isPrivate()) { /*PV*/                         \
      /* commutative, swap args */                                             \
      ret = Name##_vp(ctx, y, x);                                              \
    } else if (x.isPrivate() && y.isSecret()) { /*VS*/                         \
      ret = Name##_sv(ctx, y, x);                                              \
    } else if (x.isSecret() && y.isPrivate()) { /*SV*/                         \
      /* commutative, swap args */                                             \
      ret = Name##_sv(ctx, x, y);                                              \
    } else {                                                                   \
      SPU_THROW("Should not reach here, lhs type {}, rhs type {}", x.eltype(), \
                y.eltype());                                                   \
    }                                                                          \
    addValue(sscope, op.getResult(), std::move(ret), opts);                    \
  }

STANDARD_BINARY_COMMUTATIVE_OP_EXEC_IMPL(AddOp, kernel::hal::_add)
STANDARD_BINARY_COMMUTATIVE_OP_EXEC_IMPL(AndOp, kernel::hal::_and)
STANDARD_BINARY_COMMUTATIVE_OP_EXEC_IMPL(MulOp, kernel::hal::_mul)
STANDARD_BINARY_COMMUTATIVE_OP_EXEC_IMPL(XorOp, kernel::hal::_xor)

#undef STANDARD_BINARY_COMMUTATIVE_OP_EXEC_IMPL

#define STANDARD_SHIFT_OP_EXEC_IMPL(OpName, Name)                           \
  void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,         \
               mlir::spu::ring::OpName &op, const ExecutionOptions &opts) { \
    const auto &lhs = lookupValue(sscope, op.getLhs(), opts);               \
    if (lhs.numel() == 0) {                                                 \
      addValue(sscope, op.getResult(), lhs, opts);                          \
      return;                                                               \
    }                                                                       \
    const auto &rhs = lookupValue(sscope, op.getRhs(), opts);               \
    auto bits = kernel::hal::dump_public_as<int64_t>(sctx, rhs);            \
    Sizes sbits;                                                            \
    if (rhs.isSplat()) {                                                    \
      sbits.push_back(bits[0]);                                             \
    } else {                                                                \
      sbits.insert(sbits.begin(), bits.begin(), bits.end());                \
    }                                                                       \
    if (lhs.isSecret()) {                                                   \
      addValue(sscope, op.getResult(), Name##_s(sctx, lhs, sbits), opts);   \
    } else if (lhs.isPrivate()) {                                           \
      addValue(sscope, op.getResult(), Name##_v(sctx, lhs, sbits), opts);   \
    } else {                                                                \
      SPU_THROW("Should not reach here, lhs type {}", lhs.eltype());        \
    }                                                                       \
  }

STANDARD_SHIFT_OP_EXEC_IMPL(LShiftOp, kernel::hal::_lshift)
STANDARD_SHIFT_OP_EXEC_IMPL(RShiftOP, kernel::hal::_rshift)
STANDARD_SHIFT_OP_EXEC_IMPL(ARShiftOp, kernel::hal::_arshift)

#undef STANDARD_SHIFT_OP_EXEC_IMPL

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::P2SOp &op, const ExecutionOptions &opts) {
  const auto &in = lookupValue(sscope, op.getOperand(), opts);
  auto ret = kernel::hal::_p2s(sctx, in);
  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::TruncOp &op, const ExecutionOptions &opts) {
  auto in = lookupValue(sscope, op.getOperand(), opts);
  SignType st = op.getKnownPositive() ? SignType::Positive : SignType::Unknown;
  if (in.isSecret()) {
    addValue(sscope, op.getResult(),
             kernel::hal::_trunc_s(sctx, in, op.getBits(), st), opts);
  } else if ((in.isPrivate())) {
    addValue(sscope, op.getResult(),
             kernel::hal::_trunc_v(sctx, in, op.getBits(), st), opts);
  } else {
    SPU_THROW("Should not reach here, type {}", in.eltype());
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::BitRevOp &op, const ExecutionOptions &opts) {
  auto in = lookupValue(sscope, op.getOperand(), opts);
  if (in.isSecret()) {
    addValue(sscope, op.getResult(),
             kernel::hal::_bitrev_s(sctx, in, op.getStart(), op.getEnd()),
             opts);
  } else if (in.isPrivate()) {
    addValue(sscope, op.getResult(),
             kernel::hal::_bitrev_v(sctx, in, op.getStart(), op.getEnd()),
             opts);
  } else {
    SPU_THROW("Should not reach here, type {}", in.eltype());
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::EqualOp &op, const ExecutionOptions &opts) {
  const auto &lhs = lookupValue(sscope, op.getLhs(), opts);
  const auto &rhs = lookupValue(sscope, op.getRhs(), opts);
  MemRef ret = kernel::hal::_equal(sctx, lhs, rhs);
  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::LessOp &op, const ExecutionOptions &opts) {
  const auto &lhs = lookupValue(sscope, op.getLhs(), opts);
  const auto &rhs = lookupValue(sscope, op.getRhs(), opts);
  MemRef ret = kernel::hal::_less(sctx, lhs, rhs);
  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::DotOp &op, const ExecutionOptions &opts) {
  const auto &lhs = lookupValue(sscope, op.getLhs(), opts);
  const auto &rhs = lookupValue(sscope, op.getRhs(), opts);
  MemRef ret;
  if (lhs.isSecret() && rhs.isSecret()) {
    ret = kernel::hal::_mmul_ss(sctx, lhs, rhs);
  } else if (lhs.isPrivate() && rhs.isPrivate()) {
    ret = kernel::hal::_mmul_vv(sctx, lhs, rhs);
  } else if (lhs.isSecret() && rhs.isPublic()) {
    ret = kernel::hal::_mmul_sp(sctx, lhs, rhs);
  } else if (lhs.isPublic() && rhs.isSecret()) {
    ret = kernel::hal::_transpose(
        sctx, kernel::hal::_mmul_sp(sctx, kernel::hal::_transpose(sctx, rhs),
                                    kernel::hal::_transpose(sctx, lhs)));
  } else if (lhs.isPrivate() && rhs.isPublic()) {
    ret = kernel::hal::_mmul_vp(sctx, lhs, rhs);
  } else if (lhs.isPublic() && rhs.isPrivate()) {
    ret = kernel::hal::_transpose(
        sctx, kernel::hal::_mmul_vp(sctx, kernel::hal::_transpose(sctx, rhs),
                                    kernel::hal::_transpose(sctx, lhs)));
  } else if (lhs.isSecret() && rhs.isPrivate()) {
    ret = kernel::hal::_mmul_sv(sctx, lhs, rhs);
  } else if (lhs.isPrivate() && rhs.isSecret()) {
    ret = kernel::hal::_transpose(
        sctx, kernel::hal::_mmul_sv(sctx, kernel::hal::_transpose(sctx, rhs),
                                    kernel::hal::_transpose(sctx, lhs)));
  } else {
    SPU_THROW("Should not reach here, lhs type {}, rhs type {}", lhs.eltype(),
              rhs.eltype());
  }
  addValue(sscope, op.getResult(), std::move(ret), opts);
}

// Other ops

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::EncodeToFxpOp &op, const ExecutionOptions &opts) {
  const auto &in = lookupValue(sscope, op.getOperand(), opts);

  auto se_type = getSemanticTypeFromMlirType(op.getResult().getType());
  PtType pt_type;
  switch (in.eltype().semantic_type()) {
    case SE_I16: {
      pt_type = PT_F16;
      break;
    }
    case SE_I32: {
      pt_type = PT_F32;
      break;
    }
    case SE_I64: {
      pt_type = PT_F64;
      break;
    }
    default: {
      SPU_THROW("Unmapped semantic type {}", in.eltype());
    }
  }

  PtBufferView view(in.data(), pt_type, in.shape(), in.strides());

  auto ret = kernel::hal::_encode_fp(sctx, view, op.getFxpBits(), se_type);

  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::DecodeFromFxpOp &op,
             const ExecutionOptions &opts) {
  const auto &in = lookupValue(sscope, op.getOperand(), opts);

  auto se_type = getSemanticTypeFromMlirType(op.getResult().getType());
  auto pt_type = getPtTypeFromMlirType(op.getResult().getType());

  MemRef out(makeType<mpc::Pub2kTy>(se_type), in.shape());

  PtBufferView view(out.data(), pt_type, in.shape(), in.strides());

  kernel::hal::_decode_fp(sctx, in, &view, op.getFxpBits());

  addValue(sscope, op.getResult(), out, opts);
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::CastOp &op, const ExecutionOptions &opts) {
  auto se_type = getSemanticTypeFromMlirType(op.getResult().getType());
  const auto &in = lookupValue(sscope, op.getOperand(), opts);

  // If in width == out width, this is indeed a bitcast
  auto in_width = mlir::spu::ring::getRingWidth(op.getOperand().getType());
  auto out_width = mlir::spu::ring::getRingWidth(op.getType());

  if (in_width == out_width) {
    auto out_ty = in.eltype();
    out_ty.as<RingTy>()->set_semantic_type(se_type);

    auto ret = in.as(out_ty);
    addValue(sscope, op.getResult(), ret, opts);
  } else {
    if (in.isSecret()) {
      addValue(sscope, op.getResult(),
               kernel::hal::_ring_cast_s(sctx, in, se_type), opts);
    } else if (in.isPrivate()) {
      addValue(sscope, op.getResult(),
               kernel::hal::_ring_cast_v(sctx, in, se_type), opts);
    } else {
      SPU_THROW("Should not reach here, type {}", in.eltype());
    }
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::SecretInsertSliceOp &op,
             const ExecutionOptions &opts) {
  // Basic idea here, get a ref slice and update the whole slice..
  // Start indices
  std::vector<spu::MemRef> start_indices(op.getStartIndices().size());
  const auto &operand = lookupValue(sscope, op.getOperand(), opts);
  const auto &update = lookupValue(sscope, op.getUpdate(), opts);

  for (const auto &idx : llvm::enumerate(op.getStartIndices())) {
    start_indices[idx.index()] = lookupValue(sscope, idx.value(), opts);
  }

  addValue(sscope, op.getResult(),
           kernel::hal::DynamicUpdateSlice(sctx, operand, update, start_indices,
                                           false),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::spu::ring::SecretExtractSliceOp &op,
             const ExecutionOptions &opts) {
  // Start indices
  const auto &operand = lookupValue(sscope, op.getOperand(), opts);

  auto result_type = mlir::cast<mlir::ShapedType>(op.getType());
  std::vector<spu::MemRef> secret_indices(op.getIndices().size());

  for (const auto &idx : llvm::enumerate(op.getIndices())) {
    secret_indices[idx.index()] = lookupValue(sscope, idx.value(), opts);
  }

  auto zero_const =
      kernel::hal::_encode_int(sctx, static_cast<int64_t>(0),
                               secret_indices.front().eltype().semantic_type());

  std::vector<spu::MemRef> start_indices(result_type.getRank(), zero_const);

  for (const auto &[dim, value] :
       llvm::zip(op.getIndexingDim(), secret_indices)) {
    start_indices[dim] = value;
  }

  Sizes slice_size(result_type.getShape());

  addValue(sscope, op.getResult(),
           kernel::hal::DynamicSlice(sctx, operand, slice_size, start_indices),
           opts);
}

}  // namespace ring

// This is a hack to make template happy with namespace
using namespace ring;  // NOLINT

#include "libspu/device/utils/dispatch_template.cc.inc"

void dispatch(mlir::spu::ring::RingDialect *, SPUContext *sctx,
              SymbolScope *sscope, mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts) {
  dispatchOp<
#define GET_OP_LIST
#include "libspu/dialect/ring/IR/ops.cc.inc"
      >(executor, sctx, sscope, op, opts);
}

}  // namespace spu::device