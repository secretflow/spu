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

#include "libspu/device/pphlo/pphlo_executor.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"

#include "libspu/core/encoding.h"
#include "libspu/device/pphlo/pphlo_intrinsic_executor.h"
#include "libspu/device/pphlo/pphlo_verifier.h"
#include "libspu/dialect/pphlo_base_enums.h"
#include "libspu/dialect/pphlo_ops.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_ternary.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/control_flow.h"
#include "libspu/kernel/hlo/convolution.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/indexing.h"
#include "libspu/kernel/hlo/rand.h"
#include "libspu/kernel/hlo/reduce.h"
#include "libspu/kernel/hlo/select_and_scatter.h"
#include "libspu/kernel/hlo/shift.h"
#include "libspu/kernel/hlo/sort.h"

namespace {

template <typename T>
void convertDenseIntElementAttr(const mlir::DenseIntElementsAttr &attr,
                                T &out) {
  out.clear();
  for (const auto &v : attr.getValues<int64_t>()) {
    out.emplace_back(v);
  }
}

template <typename T>
std::string mlirObjectToString(T &&mlir_obj) {
  std::string buf;
  llvm::raw_string_ostream rss(buf);
  mlir_obj.print(rss);
  rss.flush();
  return buf;
}

std::pair<spu::PtType, bool> getPtTypeFromMlirType(mlir::Type mlir_ty) {
  mlir::pphlo::TypeTools tool;
  auto express_type = tool.getExpressedType(mlir_ty);

  if (auto ft = express_type.dyn_cast<mlir::FloatType>()) {
    switch (ft.getWidth()) {
      case 16:
        return {spu::PT_F16, false};
      case 32:
        return {spu::PT_F32, false};
      case 64:
        return {spu::PT_F64, false};
    }
  } else if (auto it = express_type.dyn_cast<mlir::IntegerType>()) {
    if (it.getWidth() == 1) {
      return {spu::PT_BOOL, false};
    }
    // In mlir, isSigned is for si[1-9][0-9]* type, isUnsigned is for
    // ui[1-9][0-9]*, i[1-9][0-9]* is signless IntegerType... So here, we only
    // check for isUnsigned, signless we treat it as signed.
    // See https://reviews.llvm.org/D72533
    switch (it.getWidth()) {
      case 8:
        return it.isUnsigned() ? std::make_pair(spu::PT_U8, false)
                               : std::make_pair(spu::PT_I8, false);
      case 16:
        return it.isUnsigned() ? std::make_pair(spu::PT_U16, false)
                               : std::make_pair(spu::PT_I16, false);
      case 32:
        return it.isUnsigned() ? std::make_pair(spu::PT_U32, false)
                               : std::make_pair(spu::PT_I32, false);
      case 64:
        return it.isUnsigned() ? std::make_pair(spu::PT_U64, false)
                               : std::make_pair(spu::PT_I64, false);
    }
  } else if (auto ct = express_type.dyn_cast<mlir::ComplexType>()) {
    if (ct.getElementType().isF32()) {
      return {spu::PT_F32, true};
    } else if (ct.getElementType().isF64()) {
      return {spu::PT_F64, true};
    }
  }

  SPU_THROW("invalid type {}", mlirObjectToString(mlir_ty));
}

spu::DataType getDtypeFromMlirType(mlir::Type mlir_ty) {
  mlir::pphlo::TypeTools tool;
  auto express_type = tool.getExpressedType(mlir_ty);
  if (auto int_ty = express_type.dyn_cast<mlir::IntegerType>()) {
    switch (int_ty.getWidth()) {
      case 1:
        return spu::DT_I1;
      case 8:
        return int_ty.isUnsigned() ? spu::DT_U8 : spu::DT_I8;
      case 16:
        return int_ty.isUnsigned() ? spu::DT_U16 : spu::DT_I16;
      case 32:
        return int_ty.isUnsigned() ? spu::DT_U32 : spu::DT_I32;
      case 64:
        return int_ty.isUnsigned() ? spu::DT_U64 : spu::DT_I64;
      default:
        SPU_THROW("unsupported int type {}", mlirObjectToString(mlir_ty));
    }
  } else if (auto flp_ty = express_type.dyn_cast<mlir::FloatType>()) {
    switch (flp_ty.getWidth()) {
      case 16:
        return spu::DT_F16;
      case 32:
        return spu::DT_F32;
      case 64:
        return spu::DT_F64;
      default:
        SPU_THROW("unsupported fp type {}", mlirObjectToString(flp_ty));
    }
  } else if (auto ct = express_type.dyn_cast<mlir::ComplexType>()) {
    if (ct.getElementType().isF32()) {
      return spu::DT_F32;
    } else if (ct.getElementType().isF64()) {
      return spu::DT_F64;
    }
  }
  SPU_THROW("invalid type {}", mlirObjectToString(mlir_ty));
}

// Convert mlir visibility to spu visibility
spu::Visibility convertVisibility(mlir::pphlo::Visibility vis) {
  switch (vis) {
    case mlir::pphlo::Visibility::VIS_PUBLIC:
      return spu::Visibility::VIS_PUBLIC;
    case mlir::pphlo::Visibility::VIS_SECRET:
      return spu::Visibility::VIS_SECRET;
  }
  SPU_THROW("Should not hit");
}

int64_t findTwoK(double in) {
  uint64_t N = 1;
  int64_t count = 0;
  while (N < in) {
    N <<= 1;
    ++count;
  }
  return --count;
}

}  // namespace

namespace spu::device::pphlo {
namespace {

void do_type_checker(mlir::Value key, const spu::Value &val,
                     const ExecutionOptions &opts) {
  if (opts.do_type_check) {
    const auto mlir_type = key.getType();
    {
      const auto &mlir_shape =
          mlir_type.dyn_cast<mlir::RankedTensorType>().getShape();
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

    // Check dtype
    mlir::pphlo::TypeTools tool;
    auto expectedType = getDtypeFromMlirType(mlir_type);
    SPU_ENFORCE(expectedType == val.dtype(), "Expected mlir_type {}, got {}",
                expectedType, val.dtype());
    if (tool.getExpressedType(mlir_type).isa<mlir::ComplexType>()) {
      SPU_ENFORCE(val.isComplex(), "Expected complex type");
    } else {
      SPU_ENFORCE(!val.isComplex());
    }

    // Check vtype
    if (tool.isMPCType<mlir::pphlo::PublicType>(mlir_type)) {
      SPU_ENFORCE(val.isPublic());
    } else if (tool.isMPCType<mlir::pphlo::SecretType>(mlir_type)) {
      SPU_ENFORCE(val.isSecret());
    } else {
      SPU_ENFORCE("Unknown vtype");
    }
  }
}

spu::Value lookupValue(SymbolScope *scope, mlir::Value key,
                       const ExecutionOptions &opts) {
  auto val = scope->lookupValue(key);
  do_type_checker(key, val, opts);
  return val;
}

void addValue(SymbolScope *scope, mlir::Value key, const spu::Value &val,
              const ExecutionOptions &opts) {
  do_type_checker(key, val, opts);
  scope->addValue(key, val);
}

void addValue(SymbolScope *scope, mlir::Value key, spu::Value &&val,
              const ExecutionOptions &opts) {
  do_type_checker(key, val, opts);
  scope->addValue(key, val);
}

void removeValue(SymbolScope *scope, mlir::Value key,
                 const ExecutionOptions &) {
  scope->removeValue(key);
}

//
#define STANDARD_UNARY_OP_EXEC_IMPL(OpName, KernelName)                 \
  void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,     \
               mlir::pphlo::OpName &op, const ExecutionOptions &opts) { \
    const auto in = lookupValue(sscope, op.getOperand(), opts);         \
    auto ret = kernel::hlo::KernelName(sctx, in);                       \
    addValue(sscope, op.getResult(), std::move(ret), opts);             \
  }

STANDARD_UNARY_OP_EXEC_IMPL(ReciprocalOp, Reciprocal)
STANDARD_UNARY_OP_EXEC_IMPL(NegOp, Neg)
STANDARD_UNARY_OP_EXEC_IMPL(ExpOp, Exp)
STANDARD_UNARY_OP_EXEC_IMPL(Expm1Op, Expm1)
STANDARD_UNARY_OP_EXEC_IMPL(LogOp, Log)
STANDARD_UNARY_OP_EXEC_IMPL(Log1pOp, Log1p)
STANDARD_UNARY_OP_EXEC_IMPL(FloorOp, Floor)
STANDARD_UNARY_OP_EXEC_IMPL(CeilOp, Ceil)
STANDARD_UNARY_OP_EXEC_IMPL(AbsOp, Abs)
STANDARD_UNARY_OP_EXEC_IMPL(LogisticOp, Logistic)
STANDARD_UNARY_OP_EXEC_IMPL(TanhOp, Tanh)
STANDARD_UNARY_OP_EXEC_IMPL(NotOp, Not)
STANDARD_UNARY_OP_EXEC_IMPL(RsqrtOp, Rsqrt)
STANDARD_UNARY_OP_EXEC_IMPL(SqrtOp, Sqrt)
STANDARD_UNARY_OP_EXEC_IMPL(RoundOp, Round_AFZ)
STANDARD_UNARY_OP_EXEC_IMPL(SineOp, Sine)
STANDARD_UNARY_OP_EXEC_IMPL(CosineOp, Cosine)

#undef STANDARD_UNARY_OP_EXEC_IMPL

#define STANDARD_BINARY_OP_EXEC_IMPL(OpName, KernelName)                      \
  void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,           \
               mlir::pphlo::OpName &op, const ExecutionOptions &opts) {       \
    addValue(                                                                 \
        sscope, op.getResult(),                                               \
        kernel::hlo::KernelName(sctx, lookupValue(sscope, op.getLhs(), opts), \
                                lookupValue(sscope, op.getRhs(), opts)),      \
        opts);                                                                \
  }

STANDARD_BINARY_OP_EXEC_IMPL(AddOp, Add)
STANDARD_BINARY_OP_EXEC_IMPL(EqualOp, Equal)
STANDARD_BINARY_OP_EXEC_IMPL(NotEqualOp, NotEqual)
STANDARD_BINARY_OP_EXEC_IMPL(LessEqualOp, LessEqual)
STANDARD_BINARY_OP_EXEC_IMPL(GreaterEqualOp, GreaterEqual)
STANDARD_BINARY_OP_EXEC_IMPL(SubtractOp, Sub)
STANDARD_BINARY_OP_EXEC_IMPL(LessOp, Less)
STANDARD_BINARY_OP_EXEC_IMPL(GreaterOp, Greater)
STANDARD_BINARY_OP_EXEC_IMPL(PowOp, Power)
STANDARD_BINARY_OP_EXEC_IMPL(MaxOp, Max)
STANDARD_BINARY_OP_EXEC_IMPL(MinOp, Min)
STANDARD_BINARY_OP_EXEC_IMPL(AndOp, And)
STANDARD_BINARY_OP_EXEC_IMPL(OrOp, Or)
STANDARD_BINARY_OP_EXEC_IMPL(XorOp, Xor)
STANDARD_BINARY_OP_EXEC_IMPL(DivOp, Div)
STANDARD_BINARY_OP_EXEC_IMPL(ShiftLeftOp, Lshift)
STANDARD_BINARY_OP_EXEC_IMPL(ShiftRightArithmeticOp, ARshift)
STANDARD_BINARY_OP_EXEC_IMPL(ShiftRightLogicalOp, Rshift)

#undef STANDARD_BINARY_OP_EXEC_IMPL

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::MulOp &op, const ExecutionOptions &opts) {
  auto smallConst = op.getRhs().getDefiningOp<mlir::pphlo::ConstantOp>();
  auto multiplier = op.getLhs();
  if (!smallConst) {
    // Try lhs
    smallConst = op.getLhs().getDefiningOp<mlir::pphlo::ConstantOp>();
    multiplier = op.getRhs();
  }

  if (smallConst && smallConst.getValue().isSplat()) {
    auto elType = smallConst.getValue().getElementType();
    if (elType.isF32() || elType.isF64()) {
      auto fValue = std::abs(smallConst.getValue()
                                 .getSplatValue<mlir::APFloat>()
                                 .convertToDouble());
      auto eps = kernel::hal::dump_public_as<float>(
          sctx, kernel::hlo::Epsilon(sctx, DT_F32))[0];

      // Amplify eps to 1/(2^(fxp_bits-2))
      // TODO: Maybe make it configurable?
      eps = eps * 4;

      if (fValue < eps && fValue > 0) {
        // Handle x * (very_small_const)
        // return truncate(x * n/N, k); n = 2^k
        // Compute N -> 1/fValue
        auto N = 1 / fValue;
        auto k = findTwoK(N);
        auto n = std::pow(2, k);

        // n/N
        auto newRhs = kernel::hlo::Constant(sctx, static_cast<float>(n) / N,
                                            smallConst.getType().getShape());
        // x*n/N
        auto kv = lookupValue(sscope, multiplier, opts);
        // To merge truncation in multiply with next k-bits one, we
        // deliberately pick the ring mul to do a mul *without* truncation
        auto mulRet = kernel::hal::_mul(sctx, kv, newRhs).setDtype(kv.dtype());
        // truncate(x*n/N, k)
        addValue(sscope, op.getResult(),
                 kernel::hal::_trunc(sctx, mulRet, sctx->getFxpBits() + k)
                     .setDtype(mulRet.dtype()),
                 opts);
        return;
      }
    }
  }

  addValue(sscope, op.getResult(),
           kernel::hlo::Mul(sctx, lookupValue(sscope, op.getLhs(), opts),
                            lookupValue(sscope, op.getRhs(), opts)),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::DotOp &op, const ExecutionOptions &opts) {
  auto ret = kernel::hlo::Dot(sctx, lookupValue(sscope, op.getLhs(), opts),
                              lookupValue(sscope, op.getRhs(), opts));

  const auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::TensorType>().getShape();

  addValue(sscope, op.getResult(), kernel::hlo::Reshape(sctx, ret, ret_shape),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::DotGeneralOp &op, const ExecutionOptions &opts) {
  auto dnum = op.getDotDimensionNumbers();
  // Should in order
  SPU_ENFORCE(dnum.getLhsBatchingDimensions().size() == 1 &&
                  dnum.getLhsContractingDimensions().size() == 1 &&
                  dnum.getLhsBatchingDimensions()[0] == 0 &&
                  dnum.getLhsContractingDimensions()[0] == 2,
              "LHS dims is not in order");
  SPU_ENFORCE(dnum.getRhsBatchingDimensions().size() == 1 &&
                  dnum.getRhsContractingDimensions().size() == 1 &&
                  dnum.getRhsBatchingDimensions()[0] == 0 &&
                  dnum.getRhsContractingDimensions()[0] == 1,
              "RHS dims is not in order");

  auto lhs = lookupValue(sscope, op.getLhs(), opts);
  auto rhs = lookupValue(sscope, op.getRhs(), opts);
  SPU_ENFORCE(lhs.shape()[0] == rhs.shape()[0], "Batch dim should equal");
  int64_t num_batch = lhs.shape()[0];

  std::vector<spu::Value> results(num_batch);
  Index lhs_slice_begin(3, 0);
  Index lhs_slice_end(lhs.shape().begin(), lhs.shape().end());
  Index rhs_slice_begin(3, 0);
  Index rhs_slice_end(rhs.shape().begin(), rhs.shape().end());
  Strides strides(lhs.shape().size(), 1);

  Shape lhs_slice_shape{lhs.shape()[1], lhs.shape()[2]};
  Shape rhs_slice_shape{rhs.shape()[1], rhs.shape()[2]};
  Shape ret_slice_shape{1, lhs.shape()[1], rhs.shape()[2]};

  for (int64_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    lhs_slice_begin[0] = batch_idx;
    lhs_slice_end[0] = batch_idx + 1;
    rhs_slice_begin[0] = batch_idx;
    rhs_slice_end[0] = batch_idx + 1;
    auto lhs_slice = kernel::hlo::Reshape(
        sctx,
        kernel::hlo::Slice(sctx, lhs, lhs_slice_begin, lhs_slice_end, strides),
        lhs_slice_shape);
    auto rhs_slice = kernel::hlo::Reshape(
        sctx,
        kernel::hlo::Slice(sctx, rhs, rhs_slice_begin, rhs_slice_end, strides),
        rhs_slice_shape);
    results[batch_idx] = kernel::hlo::Reshape(
        sctx, kernel::hlo::Dot(sctx, lhs_slice, rhs_slice), ret_slice_shape);
  }

  auto ret_type = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  auto ret = kernel::hlo::Reshape(
      sctx, kernel::hlo::Concatenate(sctx, results, 0), ret_type.getShape());

  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ConvolutionOp &op, const ExecutionOptions &opts) {
  const auto &dnums = op.getDimensionNumbers();
  const size_t num_spatial_dims = dnums.getOutputSpatialDimensions().size();
  SPU_ENFORCE(num_spatial_dims == dnums.getInputSpatialDimensions().size());
  SPU_ENFORCE(num_spatial_dims == dnums.getKernelSpatialDimensions().size());

  const auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::TensorType>().getShape();

  auto lhs = lookupValue(sscope, op.getLhs(), opts);
  auto rhs = lookupValue(sscope, op.getRhs(), opts);

  Strides window_strides(dnums.getInputSpatialDimensions().size(), 1);
  if (op.getWindowStrides().has_value()) {
    for (const auto &iter : llvm::enumerate(
             op.getWindowStrides()->getValues<int64_t>())) {  // NOLINT
      window_strides[iter.index()] = iter.value();
    }
  }

  kernel::hlo::ConvolutionConfig config;
  config.featureGroupCount = op.getFeatureGroupCount();
  config.batchGroupCount = op.getBatchGroupCount();
  config.window_strides = window_strides;
  config.inputBatchDimension = dnums.getInputBatchDimension();
  config.inputFeatureDimension = dnums.getInputFeatureDimension();
  config.inputSpatialDimensions = dnums.getInputSpatialDimensions();
  config.kernelInputFeatureDimension = dnums.getKernelInputFeatureDimension();
  config.kernelOutputFeatureDimension = dnums.getKernelOutputFeatureDimension();
  config.kernelSpatialDimensions = dnums.getKernelSpatialDimensions();
  config.outputBatchDimension = dnums.getOutputBatchDimension();
  config.outputFeatureDimension = dnums.getOutputFeatureDimension();
  config.outputSpatialDimensions = dnums.getOutputSpatialDimensions();

  SPU_ENFORCE(
      dnums.getInputSpatialDimensions().size() == 2,
      "Convolution with more than 2 spatial dimensions is not supported");

  spu::Value result =
      kernel::hlo::Convolution2D(sctx, lhs, rhs, config, ret_shape);

  addValue(sscope, op.getResult(), std::move(result), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::DynamicUpdateSliceOp &op,
             const ExecutionOptions &opts) {
  // Basic idea here, get a ref slice and update the whole slice..
  // Start indices
  std::vector<spu::Value> start_indices(op.getStartIndices().size());
  const auto &operand = lookupValue(sscope, op.getOperand(), opts);
  const auto &update = lookupValue(sscope, op.getUpdate(), opts);

  for (const auto &idx : llvm::enumerate(op.getStartIndices())) {
    start_indices[idx.index()] = lookupValue(sscope, idx.value(), opts);
  }

  addValue(
      sscope, op.getResult(),
      kernel::hlo::DynamicUpdateSlice(sctx, operand, update, start_indices),
      opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::DynamicSliceOp &op, const ExecutionOptions &opts) {
  // Start indices
  auto iter = op.getSliceSizes().getValues<int64_t>();
  Sizes slice_size{iter.begin(), iter.end()};
  const auto &operand = lookupValue(sscope, op.getOperand(), opts);
  std::vector<spu::Value> start_indices(op.getStartIndices().size());

  for (const auto &idx : llvm::enumerate(op.getStartIndices())) {
    start_indices[idx.index()] = lookupValue(sscope, idx.value(), opts);
  }

  addValue(sscope, op.getResult(),
           kernel::hlo::DynamicSlice(sctx, operand, slice_size, start_indices),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::GatherOp &op, const ExecutionOptions &opts) {
  // If input is empty, short circuit
  auto operand = lookupValue(sscope, op.getOperand(), opts);
  auto start_indices = lookupValue(sscope, op.getStartIndices(), opts);
  if (operand.numel() == 0) {
    addValue(sscope, op.getResult(), operand, opts);
    return;
  }

  const auto &output_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();

  const auto &dim_numbers = op.getDimensionNumbers();

  kernel::hlo::GatherConfig config;
  Sizes ss;
  convertDenseIntElementAttr(op.getSliceSizes(), ss);
  config.sliceSizes = ss;
  config.indexVectorDim = dim_numbers.getIndexVectorDim();
  config.offsetDims = dim_numbers.getOffsetDims();
  config.collapsedSliceDims = dim_numbers.getCollapsedSliceDims();
  config.startIndexMap = dim_numbers.getStartIndexMap();

  addValue(
      sscope, op.getResult(),
      kernel::hlo::Gather(sctx, operand, start_indices, config, output_shape),
      opts);
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::SortOp &op, const ExecutionOptions &opts) {
  auto sort_dim = op.getDimension();
  auto is_stable = op.getIsStable();
  std::vector<spu::Value> inputs(op->getNumOperands());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs[idx] = lookupValue(sscope, op->getOperand(idx), opts);
  }

  auto body_return =
      llvm::dyn_cast<mlir::pphlo::ReturnOp>(op.getComparator().back().back());
  SPU_ENFORCE(body_return, "Cannot find body return");
  SPU_ENFORCE(body_return->getNumOperands() == 1,
              "Comparator should have exactly one return");

  mlir::pphlo::TypeTools type_tools;
  auto return_vis =
      type_tools.getTypeVisibility(body_return->getOperandTypes().front());

  const spu::Visibility spu_return_vis = convertVisibility(return_vis);

  // NOTE(junfeng):
  // https://github.com/google/jax/blob/e5b2c5ea44b44439bf574cbdc0944c36b167c10c/jax/_src/numpy/lax_numpy.py#L3439
  // 'kind' is ignored in jax.numpy.sort and fixed to 'quicksort'. In order to
  // to accommodate this situation, we need to modify 'is_stable' here.
  if (is_stable && spu_return_vis == spu::Visibility::VIS_SECRET) {
    SPDLOG_WARN("only unstable sort is supported for secret returns.");
    is_stable = false;
  }

  auto ret = kernel::hlo::Sort(
      sctx, inputs, sort_dim, is_stable,
      [&](absl::Span<const spu::Value> inputs) {
        auto ret =
            runRegion(executor, sctx, sscope, op.getComparator(), inputs);
        return ret[0];
      },
      spu_return_vis);

  for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
    addValue(sscope, op->getResult(idx), std::move(ret[idx]), opts);
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::SimpleSortOp &op, const ExecutionOptions &opts) {
  auto sort_dim = op.getDimension();
  std::vector<spu::Value> inputs(op->getNumOperands());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs[idx] = lookupValue(sscope, op->getOperand(idx), opts);
  }

  kernel::hal::SortDirection direction;
  if (op.getSortDirectionAttr().getInt() ==
      static_cast<int>(mlir::pphlo::SortDirection::ASC)) {
    direction = kernel::hal::SortDirection::Ascending;
  } else if (op.getSortDirectionAttr().getInt() ==
             static_cast<int>(mlir::pphlo::SortDirection::DES)) {
    direction = kernel::hal::SortDirection::Descending;
  } else {
    SPU_THROW("Should not reach here");
  }

  auto ret = kernel::hlo::SimpleSort(sctx, inputs, sort_dim, direction);

  for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
    addValue(sscope, op->getResult(idx), ret[idx], opts);
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::SelectAndScatterOp &op,
             const ExecutionOptions &opts) {
  auto operand = lookupValue(sscope, op.getOperand(), opts);
  auto source = lookupValue(sscope, op.getSource(), opts);
  auto init_val = lookupValue(sscope, op.getInitValue(), opts);

  Shape window_shape;
  convertDenseIntElementAttr(op.getWindowDimensions(), window_shape);

  // build strides
  Strides window_strides(window_shape.size(), 1);
  if (op.getWindowStrides().has_value()) {
    convertDenseIntElementAttr(*op.getWindowStrides(), window_strides);
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});

  auto ret = kernel::hlo::SelectAndScatter(
      sctx, operand, source, init_val, window_shape, window_strides,
      window_padding,
      [&](const spu::Value &selected, const spu::Value &current) {
        auto ret = runRegion(executor, sctx, sscope, op.getSelect(),
                             {selected, current});
        return ret[0];
      },
      [&](const spu::Value &in, const spu::Value &scatter) {
        auto ret =
            runRegion(executor, sctx, sscope, op.getScatter(), {in, scatter});
        return ret[0];
      });

  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::MaxPoolScatterOp &op, const ExecutionOptions &opts) {
  auto scatter_indices = lookupValue(sscope, op.getScatterIndices(), opts);
  auto update = lookupValue(sscope, op.getUpdate(), opts);

  Shape window_shape;
  convertDenseIntElementAttr(op.getWindowDimensions().value(), window_shape);

  // build strides
  Strides window_strides(window_shape.size(), 1);
  if (op.getWindowStrides().has_value()) {
    convertDenseIntElementAttr(*op.getWindowStrides(), window_strides);
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});

  auto base_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();

  auto ret =
      kernel::hlo::MaxPoolScatter(sctx, scatter_indices, update, window_shape,
                                  base_shape, window_strides, window_padding);

  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::CaseOp &op, const ExecutionOptions &opts) {
  std::vector<kernel::hlo::BranchFcnT> branches;
  for (auto &b : op.getBranches()) {
    branches.emplace_back(
        [&]() { return runRegion(executor, sctx, sscope, b, {}); });
  }

  auto results = kernel::hlo::Case(
      sctx, lookupValue(sscope, op.getIndex(), opts), branches);

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    addValue(sscope, ret.value(), results[ret.index()], opts);
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::IfOp &op, const ExecutionOptions &opts) {
  auto conditional = lookupValue(sscope, op.getCondition(), opts);

  auto results = kernel::hlo::IfElse(
      sctx, conditional,  //
      [&]() {
        return runRegion(executor, sctx, sscope, op.getTrueBranch(), {});
      },
      [&]() {
        return runRegion(executor, sctx, sscope, op.getFalseBranch(), {});
      });

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    addValue(sscope, ret.value(), results[ret.index()], opts);
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::WhileOp &op, const ExecutionOptions &opts) {
  // First inputs vectors
  std::vector<spu::Value> inputs;
  inputs.reserve(op->getNumOperands());

  // Prepare inputs
  for (const auto operand : op->getOperands()) {
    inputs.emplace_back(lookupValue(sscope, operand, opts));
  }

  auto ret = kernel::hlo::While(
      sctx, inputs,  //
      [&](absl::Span<const spu::Value> inputs) {
        return runRegion(executor, sctx, sscope, op.getCond(), inputs)[0];
      },
      [&](absl::Span<const spu::Value> inputs) {
        return runRegion(executor, sctx, sscope, op.getBody(), inputs);
      });

  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    addValue(sscope, op->getResult(idx), std::move(ret[idx]), opts);
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::IotaOp &op, const ExecutionOptions &opts) {
  const auto &ret_type =
      op.getOutput().getType().dyn_cast<mlir::RankedTensorType>();
  const size_t numel = ret_type.getShape()[op.getIotaDimension()];

  mlir::pphlo::TypeTools type_tools;
  auto ret_el_type = type_tools.getExpressedType(ret_type);
  auto pt_type = getPtTypeFromMlirType(ret_el_type);

  spu::Value iota_ret =
      kernel::hlo::Iota(sctx, getEncodeType(pt_type.first), numel);

  if (ret_type.getShape().size() > 1) {
    // Need a broadcast
    iota_ret = kernel::hlo::Broadcast(sctx, iota_ret, ret_type.getShape(), {});
  }

  if (pt_type.second) {
    // Complex
    auto zeros = kernel::hlo::Constant(sctx, 0.0F, ret_type.getShape());
    zeros = kernel::hlo::Cast(sctx, zeros, iota_ret.vtype(), iota_ret.dtype());
    iota_ret = kernel::hlo::Complex(sctx, iota_ret, zeros);
  }

  addValue(sscope, op.getOutput(), std::move(iota_ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::RemOp &op, const ExecutionOptions &opts) {
  // FIXME: When hal has a remainder, use that
  auto lhs = lookupValue(sscope, op.getLhs(), opts);
  auto rhs = lookupValue(sscope, op.getRhs(), opts);

  auto ret = kernel::hlo::Remainder(sctx, lhs, rhs);
  addValue(sscope, op.getResult(), std::move(ret), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::TransposeOp &op, const ExecutionOptions &opts) {
  Axes permu;
  convertDenseIntElementAttr(op.getPermutation(), permu);

  addValue(sscope, op.getResult(),
           kernel::hlo::Transpose(
               sctx, lookupValue(sscope, op.getOperand(), opts), permu),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::BroadcastOp &op, const ExecutionOptions &opts) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  Axes in_dims;
  convertDenseIntElementAttr(op.getBroadcastDimensions(), in_dims);
  addValue(
      sscope, op.getResult(),
      kernel::hlo::Broadcast(sctx, lookupValue(sscope, op.getOperand(), opts),
                             to_shape, in_dims),
      opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ReshapeOp &op, const ExecutionOptions &opts) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  addValue(sscope, op.getResult(),
           kernel::hlo::Reshape(
               sctx, lookupValue(sscope, op.getOperand(), opts), to_shape),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ConcatenateOp &op, const ExecutionOptions &opts) {
  std::vector<spu::Value> values(op->getNumOperands());

  for (size_t idx = 0; idx < op->getNumOperands(); ++idx) {
    values[idx] = lookupValue(sscope, op->getOperand(idx), opts);
  }

  // set result
  addValue(sscope, op.getResult(),
           kernel::hlo::Concatenate(sctx, values, op.getDimension()), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::SliceOp &op, const ExecutionOptions &opts) {
  Index start;
  Index end;
  Strides s;
  convertDenseIntElementAttr(op.getStartIndices(), start);
  convertDenseIntElementAttr(op.getLimitIndices(), end);
  convertDenseIntElementAttr(op.getStrides(), s);
  addValue(sscope, op.getResult(),
           kernel::hlo::Slice(sctx, lookupValue(sscope, op.getOperand(), opts),
                              start, end, s),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::PadOp &op, const ExecutionOptions &opts) {
  const auto &operand = lookupValue(sscope, op.getOperand(), opts);
  const size_t operand_rank = operand.shape().size();
  const auto &padding_value = lookupValue(sscope, op.getPaddingValue(), opts);
  SPU_ENFORCE(padding_value.shape().isScalar());

  Sizes edge_padding_low;
  convertDenseIntElementAttr(op.getEdgePaddingLow(), edge_padding_low);
  SPU_ENFORCE(edge_padding_low.size() == operand_rank);

  Sizes edge_padding_high;
  convertDenseIntElementAttr(op.getEdgePaddingHigh(), edge_padding_high);
  SPU_ENFORCE(edge_padding_high.size() == operand_rank);

  Sizes interior_padding;
  convertDenseIntElementAttr(op.getInteriorPadding(), interior_padding);
  SPU_ENFORCE(interior_padding.size() == operand_rank);
  SPU_ENFORCE(std::all_of(interior_padding.begin(), interior_padding.end(),
                          [](int64_t i) { return i >= 0; }));

  addValue(sscope, op.getResult(),
           kernel::hlo::Pad(sctx, operand, padding_value, edge_padding_low,
                            edge_padding_high, interior_padding),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ReverseOp &op, const ExecutionOptions &opts) {
  Axes dims;
  convertDenseIntElementAttr(op.getDimensions(), dims);
  addValue(sscope, op.getResult(),
           kernel::hlo::Reverse(
               sctx, lookupValue(sscope, op.getOperand(), opts), dims),
           opts);
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ReduceOp &op, const ExecutionOptions &opts) {
  int64_t num_args = op->getNumOperands() / 2;
  Axes dimensions_to_reduce;
  convertDenseIntElementAttr(op.getDimensions(), dimensions_to_reduce);

  std::vector<spu::Value> input_args(num_args);
  std::vector<spu::Value> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = lookupValue(sscope, op.getInputs()[i], opts);
    init_values[i] = lookupValue(sscope, op.getInitValues()[i], opts);
  }

  bool canIgnoreInitialValue =
      std::none_of(dimensions_to_reduce.begin(), dimensions_to_reduce.end(),
                   [](int64_t d) { return d == 0; });

  std::vector<spu::Value> ret = kernel::hlo::Reduce(
      sctx, input_args, init_values, dimensions_to_reduce,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        std::vector<spu::Value> operands;
        operands.reserve(lhs.size() + rhs.size());
        operands.insert(operands.end(), lhs.begin(), lhs.end());
        operands.insert(operands.end(), rhs.begin(), rhs.end());
        return runRegion(executor, sctx, sscope, op.getBody(), operands);
      },
      canIgnoreInitialValue);

  const auto &output_shape =
      op->getResultTypes()[0].dyn_cast<mlir::RankedTensorType>().getShape();
  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    addValue(sscope, op->getResult(idx),
             kernel::hlo::Reshape(sctx, ret[idx], output_shape), opts);
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ReduceWindowOp &op, const ExecutionOptions &opts) {
  int64_t num_args = op->getNumOperands() / 2;

  std::vector<spu::Value> input_args(num_args);
  std::vector<spu::Value> init_values(num_args);

  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = lookupValue(sscope, op.getInputs()[i], opts);
    init_values[i] = lookupValue(sscope, op.getInitValues()[i], opts);
  }

  auto ret_shape = op->getResults()[0]
                       .getType()
                       .dyn_cast<mlir::RankedTensorType>()
                       .getShape();
  Shape window_shape;
  convertDenseIntElementAttr(op.getWindowDimensions(), window_shape);

  // build strides
  Strides window_strides(window_shape.size(), 1);
  if (op.getWindowStrides().has_value()) {
    convertDenseIntElementAttr(*op.getWindowStrides(),
                               window_strides);  // NOLINT
  }

  // window dilation
  Sizes window_dilations(window_shape.size(), 1);
  if (op.getWindowDilations().has_value()) {
    convertDenseIntElementAttr(*op.getWindowDilations(),
                               window_dilations);  // NOLINT
  }

  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  Sizes base_dilation(window_shape.size(), 1);

  kernel::hlo::ReduceWindowConfig config;
  config.window_shape = window_shape;
  config.window_strides = window_strides;
  config.window_dilations = window_dilations;
  config.window_padding = window_padding;
  config.base_dilations = base_dilation;

  auto rets = kernel::hlo::ReduceWindow(
      sctx, input_args, init_values, ret_shape, config,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        std::vector<spu::Value> operands;
        operands.reserve(lhs.size() + rhs.size());
        operands.insert(operands.end(), lhs.begin(), lhs.end());
        operands.insert(operands.end(), rhs.begin(), rhs.end());
        return runRegion(executor, sctx, sscope, op.getBody(), operands);
      },
      std::none_of(window_shape.begin(), window_shape.end(),
                   [](int64_t ws) { return ws == 0; }));

  for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
    addValue(sscope, op->getResults()[idx], std::move(rets[idx]), opts);
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ArgMaxOp &op, const ExecutionOptions &opts) {
  Shape window_shape;
  convertDenseIntElementAttr(op.getWindowDimensions(), window_shape);

  // build strides
  Strides window_strides(window_shape.size(), 1);
  if (op.getWindowStrides().has_value()) {
    convertDenseIntElementAttr(*op.getWindowStrides(),
                               window_strides);  // NOLINT
  }

  // window dilation
  Sizes window_dilations(window_shape.size(), 1);
  if (op.getWindowDilations().has_value()) {
    convertDenseIntElementAttr(*op.getWindowDilations(),
                               window_dilations);  // NOLINT
  }

  auto ret_shape = op->getResults()[0]
                       .getType()
                       .dyn_cast<mlir::RankedTensorType>()
                       .getShape();

  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  Sizes base_dilations(window_shape.size(), 1);

  kernel::hlo::ReduceWindowConfig config;
  config.window_shape = window_shape;
  config.window_strides = window_strides;
  config.window_dilations = window_dilations;
  config.window_padding = window_padding;
  config.base_dilations = base_dilations;

  auto ret = kernel::hlo::ArgMax(sctx, lookupValue(sscope, op.getInput(), opts),
                                 ret_shape, config);

  addValue(sscope, op.getResult(0), ret.first, opts);

  addValue(sscope, op.getResult(1), ret.second, opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::SelectOp &op, const ExecutionOptions &opts) {
  auto pred = lookupValue(sscope, op.getPred(), opts);

  auto on_true = lookupValue(sscope, op.getOnTrue(), opts);
  auto on_false = lookupValue(sscope, op.getOnFalse(), opts);

  addValue(sscope, op.getResult(),
           kernel::hlo::Select(sctx, pred, on_true, on_false), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::RngOp &op, const ExecutionOptions &opts) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  addValue(
      sscope, op.getResult(),
      kernel::hlo::Uniform_rand(sctx, lookupValue(sscope, op.getA(), opts),
                                lookupValue(sscope, op.getB(), opts), to_shape),
      opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ConvertOp &op, const ExecutionOptions &opts) {
  mlir::pphlo::TypeTools tool;
  auto dst_dtype = getDtypeFromMlirType(op.getType());
  auto dst_vtype = tool.isMPCType<mlir::pphlo::PublicType>(op.getType())
                       ? VIS_PUBLIC
                       : VIS_SECRET;
  auto in = lookupValue(sscope, op.getOperand(), opts);

  auto from_type = tool.getExpressedType(op.getOperand().getType());
  auto to_type = tool.getExpressedType(op.getType());

  auto casted = kernel::hlo::Cast(sctx, in, dst_vtype, dst_dtype);
  if (!from_type.isa<mlir::ComplexType>() && to_type.isa<mlir::ComplexType>()) {
    auto imag = kernel::hlo::Imag(sctx, casted);
    casted = kernel::hlo::Complex(sctx, casted, imag);
  }

  addValue(sscope, op.getResult(), casted, opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::PreferAOp &op, const ExecutionOptions &opts) {
  auto in = lookupValue(sscope, op.getOperand(), opts);
  if (sctx->config().protocol() == ProtocolKind::CHEETAH) {
    // NOTE(juhou): For 2PC, MulAB uses COT which is efficient and accurate than
    // MulAA that needs HE. Thus we just by-pass the PreferAOp for 2PC.
    addValue(sscope, op.getResult(), in, opts);
    return;
  }
  auto k0 = kernel::hlo::Cast(sctx, kernel::hlo::Constant(sctx, 0, in.shape()),
                              VIS_PUBLIC, in.dtype());
  addValue(sscope, op.getResult(), kernel::hlo::Add(sctx, in, k0), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::SignOp &op, const ExecutionOptions &opts) {
  auto in = lookupValue(sscope, op.getOperand(), opts);
  addValue(sscope, op.getResult(), kernel::hlo::Sign(sctx, in), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::BitcastConvertOp &op, const ExecutionOptions &opts) {
  const auto &in_type =
      op.getOperand().getType().dyn_cast<mlir::RankedTensorType>();
  const auto &out_type =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();

  // bitcast should not change total #bytes, so if sizeof(in_t) !=
  // sizeof(out_t) will result to a shape change, thus it's enough to just
  // ensure in_shape == out_shape
  SPU_ENFORCE(in_type.getShape() == out_type.getShape(),
              "bitcast with different size is not supported yet");

  addValue(
      sscope, op.getResult(),
      kernel::hlo::Bitcast(sctx, lookupValue(sscope, op.getOperand(), opts),
                           getDtypeFromMlirType(out_type)),
      opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ConstantOp &op, const ExecutionOptions &opts) {
  const auto &val = op.getValue();
  const auto &dea = val.dyn_cast<mlir::DenseElementsAttr>();
  const auto &type = val.getType().dyn_cast<mlir::RankedTensorType>();
  const Shape &dst_shape = type.getShape();
  const auto &pt_type = getPtTypeFromMlirType(type.getElementType());

  // For 1-bit type, MLIR buffer is either 0 or 255
  // See
  // https://github.com/llvm/llvm-project/blob/3696941dae5cc5bb379c50eae6190e29f7edbbb1/mlir/include/mlir/IR/BuiltinAttributes.h#L188
  // We need to normalize the value to 0,1
  if (dea.getElementType().isInteger(1)) {
    SPU_ENFORCE(pt_type.second == false);
    if (dea.isSplat()) {
      addValue(
          sscope, op.getResult(),
          kernel::hlo::Constant(sctx, dea.getSplatValue<bool>(), dst_shape),
          opts);
    } else {
      std::vector<uint8_t> buf(type.getNumElements());
      for (const auto &v : llvm::enumerate(dea.getValues<bool>())) {
        buf[v.index()] = static_cast<uint8_t>(v.value());
      }
      PtBufferView view(reinterpret_cast<const bool *>(buf.data()),
                        pt_type.first, dst_shape,
                        makeCompactStrides(dst_shape));

      addValue(sscope, op.getResult(),
               kernel::hlo::Constant(sctx, view, dst_shape), opts);
    }
  } else {
    if (!pt_type.second) {
      // Real numbers
      PtBufferView view(
          dea.getRawData().data(), pt_type.first,
          dea.isSplat() ? Shape() : dst_shape,
          dea.isSplat() ? Strides() : makeCompactStrides(dst_shape));

      addValue(sscope, op.getResult(),
               kernel::hlo::Constant(sctx, view, dst_shape), opts);
    } else {
      // Complex constant
      // real view
      auto cs = makeCompactStrides(dst_shape);
      if (!cs.empty()) {
        for (auto &s : cs) {
          s *= 2;
        }
      }
      PtBufferView real_view(dea.getRawData().data(), pt_type.first,
                             dea.isSplat() ? Shape() : dst_shape,
                             dea.isSplat() ? Strides() : cs);
      PtBufferView imag_view(dea.getRawData().data() + SizeOf(pt_type.first),
                             pt_type.first, dea.isSplat() ? Shape() : dst_shape,
                             dea.isSplat() ? Strides() : cs);

      auto real = kernel::hlo::Constant(sctx, real_view, dst_shape);
      auto imag = kernel::hlo::Constant(sctx, imag_view, dst_shape);

      addValue(sscope, op.getResult(), kernel::hlo::Complex(sctx, real, imag),
               opts);
    }
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::EpsilonOp &op, const ExecutionOptions &opts) {
  auto e = kernel::hlo::Epsilon(sctx, getDtypeFromMlirType(op.getType()));
  auto shape =
      op->getResultTypes()[0].dyn_cast<mlir::RankedTensorType>().getShape();
  addValue(sscope, op.getResult(), kernel::hlo::Broadcast(sctx, e, shape, {}),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ClampOp &op, const ExecutionOptions &opts) {
  addValue(sscope, op.getResult(),
           kernel::hlo::Clamp(sctx, lookupValue(sscope, op.getOperand(), opts),
                              lookupValue(sscope, op.getMin(), opts),
                              lookupValue(sscope, op.getMax(), opts)),
           opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::CustomCallOp &op, const ExecutionOptions &opt) {
  std::vector<Value> inputs(op->getNumOperands());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs[idx] = lookupValue(sscope, op->getOperand(idx), opt);
  }
  auto ret = intrinsic_dispatcher(sctx, op.getCallTargetName(), inputs);

  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    addValue(sscope, op->getResult(idx), std::move(ret[idx]), opt);
  }
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::DbgPrintOp &op, const ExecutionOptions &opts) {
  kernel::hal::dbg_print(sctx, lookupValue(sscope, op.getOperand(), opts));
}

void execute(OpExecutor *, SPUContext *, SymbolScope *sscope,
             mlir::pphlo::FreeOp &op, const ExecutionOptions &opts) {
  if (opts.do_parallel) {
    // Think about the following case
    // %a = def
    // use(%a)
    // use(%a)
    // free(%a)
    // Here free is also a consider a use...so under parallel execution free
    // will be invoked once a is defined.
    // This will make %a randomly deallocated after defined.
    // FreeOp has an implicit requirement that it needs to be invoked after all
    // other uses are done.
    // FIXME(xiaochen): Enable this...
    return;
  }
  removeValue(sscope, op.getOperand(), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::RealOp &op, const ExecutionOptions &opts) {
  auto v = lookupValue(sscope, op.getOperand(), opts);
  addValue(sscope, op.getResult(), kernel::hlo::Real(sctx, v), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ImagOp &op, const ExecutionOptions &opts) {
  auto v = lookupValue(sscope, op.getOperand(), opts);
  addValue(sscope, op.getResult(), kernel::hlo::Imag(sctx, v), opts);
}

void execute(OpExecutor *, SPUContext *sctx, SymbolScope *sscope,
             mlir::pphlo::ComplexOp &op, const ExecutionOptions &opts) {
  auto r = lookupValue(sscope, op.getLhs(), opts);
  auto i = lookupValue(sscope, op.getRhs(), opts);
  addValue(sscope, op.getResult(), kernel::hlo::Complex(sctx, r, i), opts);
}

#define DEFINE_UNIMPLEMENTED_OP(OpName)                           \
  void execute(OpExecutor *, SPUContext *, SymbolScope *,         \
               mlir::pphlo::OpName &, const ExecutionOptions &) { \
    SPU_THROW("Lowered op should not occur at backend");          \
  }

DEFINE_UNIMPLEMENTED_OP(ReturnOp)

#undef DEFINE_UNIMPLEMENTED_OP

}  // namespace

template <typename OpT, typename... MoreOpT>
static bool hasKernelImpl(mlir::Operation &op) {
  if (auto casted = llvm::dyn_cast<OpT>(op)) {
    return true;
  } else {
    if constexpr (!sizeof...(MoreOpT)) {
      return false;
    } else {
      return hasKernelImpl<MoreOpT...>(op);
    }
  }
}

bool PPHloExecutor::hasKernel(mlir::Operation &op) const {
  return hasKernelImpl<
#define GET_OP_LIST
#include "libspu/dialect/pphlo_ops.cc.inc"
      >(op);
}

template <typename OpT, typename... MoreOpT>
static void dispatchOp(OpExecutor *executor, SPUContext *sctx,
                       SymbolScope *sscope, mlir::Operation &op,
                       const ExecutionOptions &opts) {
  if (auto casted = llvm::dyn_cast<OpT>(op)) {
    // Execute op
    {
      const auto fn_name = op.getName().getStringRef().str();
      SPU_TRACE_ACTION(GET_TRACER(sctx), sctx->lctx(), (TR_HLO | TR_LAR),
                       ~TR_HLO, fn_name);
      execute(executor, sctx, sscope, casted, opts);
    }

    // currently we only support config verifier statically.
    constexpr bool kEnableXlaVerifier = false;
    if (kEnableXlaVerifier) {
      PPHloVerifier verifier(sctx);
      // handle mixed (int, fxp) multiplication
      if constexpr (std::is_same_v<OpT, mlir::pphlo::MulOp> or
                    std::is_same_v<OpT, mlir::pphlo::DotOp> or
                    std::is_same_v<OpT, mlir::pphlo::DotGeneralOp>) {
        spu::Value lhs = sscope->lookupValue(casted.getLhs());
        spu::Value rhs = sscope->lookupValue(casted.getRhs());
        spu::Value ret = sscope->lookupValue(casted.getResult());
        mlir::pphlo::TypeTools type_tool;
        auto lhs_type = type_tool.getExpressedType(casted.getLhs().getType());
        auto rhs_type = type_tool.getExpressedType(casted.getRhs().getType());
        auto ret_type =
            type_tool.getExpressedType(casted.getResult().getType());

        if (lhs_type != ret_type) {
          lhs = kernel::hlo::Cast(sctx, lhs, lhs.vtype(), ret.dtype());
        }
        if (rhs_type != ret_type) {
          rhs = kernel::hlo::Cast(sctx, rhs, rhs.vtype(), ret.dtype());
        }

        verifier.verify(casted, {lhs, rhs}, {ret});
      } else {
        // Collect inputs
        std::vector<spu::Value> ins;
        for (auto operand : op.getOperands()) {
          ins.emplace_back(sscope->lookupValue(operand));
        }
        std::vector<spu::Value> outs;
        for (auto operand : op.getResults()) {
          outs.emplace_back(sscope->lookupValue(operand));
        }

        verifier.verify(casted, ins, outs);
      }
    }
  } else {
    if constexpr (!sizeof...(MoreOpT)) {
      SPU_THROW("Unhandled mlir op {} at {}", mlirObjectToString(op),
                mlirObjectToString(op.getLoc()));
    } else {
      dispatchOp<MoreOpT...>(executor, sctx, sscope, op, opts);
    }
  }
}

void PPHloExecutor::runKernelImpl(SPUContext *sctx, SymbolScope *sscope,
                                  mlir::Operation &op,
                                  const ExecutionOptions &opts) {
  if (opts.do_log_execution) {
    SPDLOG_INFO("PPHLO {}", mlirObjectToString(op));
  }
  dispatchOp<
#define GET_OP_LIST
#include "libspu/dialect/pphlo_ops.cc.inc"
      >(this, sctx, sscope, op, opts);
}

void PPHloExecutor::checkType(mlir::Type, const spu::Value &) const {}

}  // namespace spu::device::pphlo
