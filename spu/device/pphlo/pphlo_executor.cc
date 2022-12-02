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

#include "spu/device/pphlo/pphlo_executor.h"

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"

#include "spu/dialect/pphlo_base_enums.h"
#include "spu/dialect/pphlo_ops.h"
#include "spu/kernel/hlo/basic_binary.h"
#include "spu/kernel/hlo/basic_ternary.h"
#include "spu/kernel/hlo/basic_unary.h"
#include "spu/kernel/hlo/casting.h"
#include "spu/kernel/hlo/const.h"
#include "spu/kernel/hlo/control_flow.h"
#include "spu/kernel/hlo/convolution.h"
#include "spu/kernel/hlo/dynamic_slice.h"
#include "spu/kernel/hlo/geometrical.h"
#include "spu/kernel/hlo/indexing.h"
#include "spu/kernel/hlo/rand.h"
#include "spu/kernel/hlo/reduce.h"
#include "spu/kernel/hlo/select_and_scatter.h"
#include "spu/kernel/hlo/shift.h"
#include "spu/kernel/hlo/sort.h"

namespace {

std::vector<int64_t>
convertDenseIntElementAttr(const mlir::DenseIntElementsAttr &attr) {
  std::vector<int64_t> ret;

  for (const auto &v : attr.getValues<int64_t>()) {
    ret.emplace_back(v);
  }

  return ret;
}

template <typename T>
std::string mlirObjectToString(T &&mlir_obj) {
  std::string buf;
  llvm::raw_string_ostream rss(buf);
  mlir_obj.print(rss);
  rss.flush();
  return buf;
}

spu::PtType getPtTypeFromMlirType(mlir::Type mlir_ty) {
  mlir::pphlo::TypeTools tool;
  auto express_type = tool.getExpressedType(mlir_ty);

  if (auto ft = express_type.dyn_cast<mlir::FloatType>()) {
    switch (ft.getWidth()) {
    case 32:
      return spu::PT_F32;
    case 64:
      return spu::PT_F64;
    }
  } else if (auto it = express_type.dyn_cast<mlir::IntegerType>()) {
    if (it.getWidth() == 1) {
      return spu::PT_BOOL;
    }
    // In mlir, isSigned is for si[1-9][0-9]* type, isUnsigned is for
    // ui[1-9][0-9]*, i[1-9][0-9]* is signless IntegerType... So here, we only
    // check for isUnsigned, signless we treat it as signed.
    // See https://reviews.llvm.org/D72533
    switch (it.getWidth()) {
    case 8:
      return it.isUnsigned() ? spu::PT_U8 : spu::PT_I8;
    case 16:
      return it.isUnsigned() ? spu::PT_U16 : spu::PT_I16;
    case 32:
      return it.isUnsigned() ? spu::PT_U32 : spu::PT_I32;
    case 64:
      return it.isUnsigned() ? spu::PT_U64 : spu::PT_I64;
    }
  }
  YACL_THROW("invalid type {}", mlirObjectToString(mlir_ty));
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
      YACL_THROW("unsupported int type {}", mlirObjectToString(mlir_ty));
    }
  } else if (auto flp_ty = express_type.dyn_cast<mlir::FloatType>()) {
    return spu::DT_FXP;
  }
  YACL_THROW("invalid type {}", mlirObjectToString(mlir_ty));
}

// Convert mlir visibility to spu visibility
spu::Visibility convertVisibility(mlir::pphlo::Visibility vis) {
  switch (vis) {
  case mlir::pphlo::Visibility::VIS_PUBLIC:
    return spu::Visibility::VIS_PUBLIC;
  case mlir::pphlo::Visibility::VIS_SECRET:
    return spu::Visibility::VIS_SECRET;
  }
  YACL_THROW("Should not hit");
}

} // namespace

namespace spu::device::pphlo {
namespace {

const spu::Value &lookupValue(SymbolScope *scope, mlir::Value key,
                              const ExecutionOptions &opts) {
  const auto &val = scope->lookupValue(key);

  if (opts.do_type_check) {
    const auto mlir_type = key.getType();
    {
      const auto &mlir_shape =
          mlir_type.dyn_cast<mlir::RankedTensorType>().getShape();
      const auto &spu_shape = val.shape();

      YACL_ENFORCE(mlir_shape.size() == spu_shape.size(),
                   "Runtime shape mismatch, expected={}, got={}",
                   fmt::join(mlir_shape, "x"), fmt::join(spu_shape, "x"));

      for (size_t idx = 0; idx < mlir_shape.size(); ++idx) {
        YACL_ENFORCE(mlir_shape[idx] == spu_shape[idx],
                     "Runtime shape mismatch at dim {}, expected={}, got={}",
                     idx, fmt::join(mlir_shape, "x"),
                     fmt::join(spu_shape, "x"));
      }
    }

    // Check dtype
    mlir::pphlo::TypeTools tool;
    auto expectedType = getDtypeFromMlirType(mlir_type);
    YACL_ENFORCE(expectedType == val.dtype(), "Expected mlir_type {}, got {}",
                 expectedType, val.dtype());

    // Check vtype
    if (tool.isMPCType<mlir::pphlo::PublicType>(mlir_type)) {
      YACL_ENFORCE(val.isPublic());
    } else if (tool.isMPCType<mlir::pphlo::SecretType>(mlir_type)) {
      YACL_ENFORCE(val.isSecret());
    } else {
      YACL_ENFORCE("Unknown vtype");
    }
  }
  return val;
}

//
#define STANDARD_UNARY_OP_EXEC_IMPL(OpName, KernelName)                        \
  void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,    \
               mlir::pphlo::OpName &op, const ExecutionOptions &opts) {        \
    const auto in = lookupValue(sscope, op.getOperand(), opts);                \
    auto ret = kernel::hlo::KernelName(hctx, in);                              \
    sscope->addValue(op.getResult(), std::move(ret));                          \
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

#undef STANDARD_UNARY_OP_EXEC_IMPL

#define STANDARD_BINARY_OP_EXEC_IMPL(OpName, KernelName)                       \
  void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,    \
               mlir::pphlo::OpName &op, const ExecutionOptions &opts) {        \
    sscope->addValue(                                                          \
        op.getResult(),                                                        \
        kernel::hlo::KernelName(hctx, lookupValue(sscope, op.lhs(), opts),     \
                                lookupValue(sscope, op.rhs(), opts)));         \
  }

STANDARD_BINARY_OP_EXEC_IMPL(AddOp, Add)
STANDARD_BINARY_OP_EXEC_IMPL(EqualOp, Equal)
STANDARD_BINARY_OP_EXEC_IMPL(NotEqualOp, NotEqual)
STANDARD_BINARY_OP_EXEC_IMPL(LessEqualOp, LessEqual)
STANDARD_BINARY_OP_EXEC_IMPL(GreaterEqualOp, GreaterEqual)
STANDARD_BINARY_OP_EXEC_IMPL(SubtractOp, Sub)
STANDARD_BINARY_OP_EXEC_IMPL(LessOp, Less)
STANDARD_BINARY_OP_EXEC_IMPL(GreaterOp, Greater)
STANDARD_BINARY_OP_EXEC_IMPL(MulOp, Mul)
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

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::DotOp &op, const ExecutionOptions &opts) {
  auto ret = kernel::hlo::Dot(hctx, lookupValue(sscope, op.lhs(), opts),
                              lookupValue(sscope, op.rhs(), opts));

  const auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::TensorType>().getShape();

  sscope->addValue(op.getResult(), kernel::hlo::Reshape(hctx, ret, ret_shape));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::DotGeneralOp &op, const ExecutionOptions &opts) {
  auto dnum = op.dot_dimension_numbers();
  // Should in order
  YACL_ENFORCE(dnum.getLhsBatchingDimensions().size() == 1 &&
                   dnum.getLhsContractingDimensions().size() == 1 &&
                   dnum.getLhsBatchingDimensions()[0] == 0 &&
                   dnum.getLhsContractingDimensions()[0] == 2,
               "LHS dims is not in order");
  YACL_ENFORCE(dnum.getRhsBatchingDimensions().size() == 1 &&
                   dnum.getRhsContractingDimensions().size() == 1 &&
                   dnum.getRhsBatchingDimensions()[0] == 0 &&
                   dnum.getRhsContractingDimensions()[0] == 1,
               "RHS dims is not in order");

  auto lhs = lookupValue(sscope, op.lhs(), opts);
  auto rhs = lookupValue(sscope, op.rhs(), opts);
  YACL_ENFORCE(lhs.shape()[0] == rhs.shape()[0], "Batch dim should equal");
  int64_t num_batch = lhs.shape()[0];

  std::vector<spu::Value> results(num_batch);
  std::vector<int64_t> lhs_slice_begin(3, 0);
  std::vector<int64_t> lhs_slice_end = lhs.shape();
  std::vector<int64_t> rhs_slice_begin(3, 0);
  std::vector<int64_t> rhs_slice_end = rhs.shape();
  std::vector<int64_t> strides(lhs.shape().size(), 1);

  std::vector<int64_t> lhs_slice_shape{lhs.shape()[1], lhs.shape()[2]};
  std::vector<int64_t> rhs_slice_shape{rhs.shape()[1], rhs.shape()[2]};
  std::vector<int64_t> ret_slice_shape{1, lhs.shape()[1], rhs.shape()[2]};

  for (int64_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    lhs_slice_begin[0] = batch_idx;
    lhs_slice_end[0] = batch_idx + 1;
    rhs_slice_begin[0] = batch_idx;
    rhs_slice_end[0] = batch_idx + 1;
    auto lhs_slice = kernel::hlo::Reshape(
        hctx,
        kernel::hlo::Slice(hctx, lhs, lhs_slice_begin, lhs_slice_end, strides),
        lhs_slice_shape);
    auto rhs_slice = kernel::hlo::Reshape(
        hctx,
        kernel::hlo::Slice(hctx, rhs, rhs_slice_begin, rhs_slice_end, strides),
        rhs_slice_shape);
    results[batch_idx] = kernel::hlo::Reshape(
        hctx, kernel::hlo::Dot(hctx, lhs_slice, rhs_slice), ret_slice_shape);
  }

  auto ret_type = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  auto ret = kernel::hlo::Reshape(
      hctx, kernel::hlo::Concatenate(hctx, results, 0), ret_type.getShape());

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ConvolutionOp &op, const ExecutionOptions &opts) {
  const auto &dnums = op.dimension_numbers();
  const size_t num_spatial_dims = dnums.getOutputSpatialDimensions().size();
  YACL_ENFORCE(num_spatial_dims == dnums.getInputSpatialDimensions().size());
  YACL_ENFORCE(num_spatial_dims == dnums.getKernelSpatialDimensions().size());

  const auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::TensorType>().getShape();

  auto lhs = lookupValue(sscope, op.lhs(), opts);
  auto rhs = lookupValue(sscope, op.rhs(), opts);

  std::vector<int64_t> window_strides(dnums.getInputSpatialDimensions().size(),
                                      1);
  if (op.window_strides().has_value()) {
    for (const auto &iter :
         llvm::enumerate(op.window_strides()->getValues<int64_t>())) {
      window_strides[iter.index()] = iter.value();
    }
  }

  kernel::hlo::ConvolutionConfig config;
  config.featureGroupCount = op.feature_group_count();
  config.batchGroupCount = op.batch_group_count();
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

  spu::Value result;

  if (dnums.getInputSpatialDimensions().size() == 2) {
    result = kernel::hlo::Convolution2D(hctx, lhs, rhs, config, ret_shape);
  } else {
    result = kernel::hlo::Convolution(hctx, lhs, rhs, config, ret_shape);
  }

  sscope->addValue(op.getResult(), std::move(result));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::DynamicUpdateSliceOp &op,
             const ExecutionOptions &opts) {
  // Basic idea here, get a ref slice and update the whole slice..
  // Start indicies
  std::vector<spu::Value> start_indicies(op.start_indices().size());
  const auto &operand = lookupValue(sscope, op.operand(), opts);
  const auto &update = lookupValue(sscope, op.update(), opts);

  for (const auto &idx : llvm::enumerate(op.start_indices())) {
    start_indicies[idx.index()] = lookupValue(sscope, idx.value(), opts);
  }

  sscope->addValue(op.getResult(), kernel::hlo::DynamicUpdateSlice(
                                       hctx, operand, update, start_indicies));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::DynamicSliceOp &op, const ExecutionOptions &opts) {
  // Start indicies
  auto iter = op.slice_sizes().getValues<int64_t>();
  std::vector<int64_t> slice_size{iter.begin(), iter.end()};
  const auto &operand = lookupValue(sscope, op.operand(), opts);
  std::vector<spu::Value> start_indicies(op.start_indices().size());

  for (const auto &idx : llvm::enumerate(op.start_indices())) {
    start_indicies[idx.index()] = lookupValue(sscope, idx.value(), opts);
  }

  sscope->addValue(
      op.getResult(),
      kernel::hlo::DynamicSlice(hctx, operand, slice_size, start_indicies));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::GatherOp &op, const ExecutionOptions &opts) {
  // If input is empty, short circuit
  auto operand = lookupValue(sscope, op.operand(), opts);
  auto start_indicies = lookupValue(sscope, op.start_indices(), opts);
  if (operand.numel() == 0) {
    sscope->addValue(op.getResult(), operand);
    return;
  }

  const auto &output_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();

  const auto &dim_numbers = op.dimension_numbers();

  kernel::hlo::GatherConfig config;
  auto ss = convertDenseIntElementAttr(op.slice_sizes());
  config.sliceSizes = ss;
  config.indexVectorDim = dim_numbers.getIndexVectorDim();
  config.offsetDims = dim_numbers.getOffsetDims();
  config.collapsedSliceDims = dim_numbers.getCollapsedSliceDims();
  config.startIndexMap = dim_numbers.getStartIndexMap();

  sscope->addValue(
      op.getResult(),
      kernel::hlo::Gather(hctx, operand, start_indicies, config, output_shape));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::SortOp &op, const ExecutionOptions &opts) {
  auto sort_dim = op.dimension();
  auto is_stable = op.is_stable();
  std::vector<spu::Value> inputs(op->getNumOperands());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs[idx] = lookupValue(sscope, op->getOperand(idx), opts);
  }

  auto body_return =
      llvm::dyn_cast<mlir::pphlo::ReturnOp>(op.comparator().back().back());
  YACL_ENFORCE(body_return, "Cannot find body return");
  YACL_ENFORCE(body_return->getNumOperands() == 1,
               "Comparator should have exactly one return");

  mlir::pphlo::TypeTools type_tools;
  auto return_vis =
      type_tools.getTypeVisibility(body_return->getOperandTypes().front());

  auto ret = kernel::hlo::Sort(
      hctx, inputs, sort_dim, is_stable,
      [&](absl::Span<const spu::Value> inputs) {
        auto ret = runRegion(executor, hctx, sscope, op.comparator(), inputs);
        return ret[0];
      },
      convertVisibility(return_vis));

  for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
    sscope->addValue(op->getResult(idx), std::move(ret[idx]));
  }
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::SelectAndScatterOp &op,
             const ExecutionOptions &opts) {
  auto operand = lookupValue(sscope, op.operand(), opts);
  auto source = lookupValue(sscope, op.source(), opts);
  auto init_val = lookupValue(sscope, op.init_value(), opts);

  auto window_shape = convertDenseIntElementAttr(op.window_dimensions());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().has_value()) {
    window_strides = convertDenseIntElementAttr(*op.window_strides());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().has_value()) {
    const auto v = *op.padding();

    YACL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  // auto ret = kernel::hlo::SelectAndScatterNaive(
  auto ret = kernel::hlo::SelectAndScatterExpanded(
      hctx, operand, source, init_val, window_shape, window_strides,
      window_padding,
      [&](const spu::Value &selected, const spu::Value &current) {
        auto ret =
            runRegion(executor, hctx, sscope, op.select(), {selected, current});
        return ret[0];
      },
      [&](const spu::Value &in, const spu::Value &scatter) {
        auto ret =
            runRegion(executor, hctx, sscope, op.scatter(), {in, scatter});
        return ret[0];
      });

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::MaxPoolScatterOp &op, const ExecutionOptions &opts) {
  auto scatter_indices = lookupValue(sscope, op.scatter_indices(), opts);
  auto update = lookupValue(sscope, op.update(), opts);

  auto window_shape =
      convertDenseIntElementAttr(op.window_dimensions().value());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().has_value()) {
    window_strides = convertDenseIntElementAttr(*op.window_strides());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().has_value()) {
    const auto v = *op.padding();

    YACL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  auto base_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();

  auto ret =
      kernel::hlo::MaxPoolScatter(hctx, scatter_indices, update, window_shape,
                                  base_shape, window_strides, window_padding);

  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::IfOp &op, const ExecutionOptions &opts) {
  auto conditional = lookupValue(sscope, op.condition(), opts);

  auto results = kernel::hlo::IfElse(
      hctx, conditional, //
      [&]() { return runRegion(executor, hctx, sscope, op.true_branch(), {}); },
      [&]() {
        return runRegion(executor, hctx, sscope, op.false_branch(), {});
      });

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    sscope->addValue(ret.value(), results[ret.index()]);
  }
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::WhileOp &op, const ExecutionOptions &opts) {
  // First inputs vectors
  std::vector<spu::Value> inputs;
  inputs.reserve(op->getNumOperands());

  // Prepare inputs
  for (const auto operand : op->getOperands()) {
    inputs.emplace_back(lookupValue(sscope, operand, opts));
  }

  auto ret = kernel::hlo::While(
      hctx, inputs, //
      [&](absl::Span<const spu::Value> inputs) {
        return runRegion(executor, hctx, sscope, op.cond(), inputs)[0];
      },
      [&](absl::Span<const spu::Value> inputs) {
        return runRegion(executor, hctx, sscope, op.body(), inputs);
      });

  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    sscope->addValue(op->getResult(idx), std::move(ret[idx]));
  }
}

#define DISPATCH_ALL_NONE_BOOL_PT_TYPES(PT_TYPE, NAME, ...)                    \
  [&] {                                                                        \
    switch (PT_TYPE) {                                                         \
      __CASE_PT_TYPE(spu::PT_I8, NAME, __VA_ARGS__)                            \
      __CASE_PT_TYPE(spu::PT_U8, NAME, __VA_ARGS__)                            \
      __CASE_PT_TYPE(spu::PT_I16, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_U16, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_I32, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_U32, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_I64, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_U64, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_F32, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_F64, NAME, __VA_ARGS__)                           \
    default:                                                                   \
      YACL_THROW("{} not implemented for pt_type={}", #NAME, PT_TYPE);         \
    }                                                                          \
  }()

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::IotaOp &op, const ExecutionOptions &opts) {
  const auto &ret_type =
      op.output().getType().dyn_cast<mlir::RankedTensorType>();
  const size_t numel = ret_type.getShape()[op.iota_dimension()];

  mlir::pphlo::TypeTools type_tools;
  auto ret_el_type = type_tools.getExpressedType(ret_type);
  auto pt_type = getPtTypeFromMlirType(ret_el_type);

  spu::Value iota_ret;
  DISPATCH_ALL_NONE_BOOL_PT_TYPES(pt_type, "_", [&] {
    iota_ret = kernel::hlo::Iota<ScalarT>(hctx, numel, VIS_PUBLIC);
  });

  if (ret_type.getShape().size() > 1) {
    // Need a broadcast
    iota_ret = kernel::hlo::Broadcast(hctx, iota_ret, ret_type.getShape(), {});
  }

  sscope->addValue(op.output(), std::move(iota_ret));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::RemOp &op, const ExecutionOptions &opts) {
  // FIXME: When hal has a remainder, use that
  auto lhs = lookupValue(sscope, op.lhs(), opts);
  auto rhs = lookupValue(sscope, op.rhs(), opts);

  auto ret = kernel::hlo::Remainder(hctx, lhs, rhs);
  sscope->addValue(op.getResult(), std::move(ret));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::TransposeOp &op, const ExecutionOptions &opts) {
  sscope->addValue(
      op.getResult(),
      kernel::hlo::Transpose(hctx, lookupValue(sscope, op.getOperand(), opts),
                             convertDenseIntElementAttr(op.permutation())));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::BroadcastOp &op, const ExecutionOptions &opts) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  sscope->addValue(op.getResult(),
                   kernel::hlo::Broadcast(
                       hctx, lookupValue(sscope, op.getOperand(), opts),
                       to_shape,
                       convertDenseIntElementAttr(op.broadcast_dimensions())));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ReshapeOp &op, const ExecutionOptions &opts) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  sscope->addValue(
      op.getResult(),
      kernel::hlo::Reshape(hctx, lookupValue(sscope, op.getOperand(), opts),
                           to_shape));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ConcatenateOp &op, const ExecutionOptions &opts) {
  std::vector<spu::Value> values(op->getNumOperands());

  for (size_t idx = 0; idx < op->getNumOperands(); ++idx) {
    values[idx] = lookupValue(sscope, op->getOperand(idx), opts);
  }

  // set result
  sscope->addValue(op.getResult(),
                   kernel::hlo::Concatenate(hctx, values, op.dimension()));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::SliceOp &op, const ExecutionOptions &opts) {
  sscope->addValue(
      op.getResult(),
      kernel::hlo::Slice(hctx, lookupValue(sscope, op.getOperand(), opts),
                         convertDenseIntElementAttr(op.start_indices()),
                         convertDenseIntElementAttr(op.limit_indices()),
                         convertDenseIntElementAttr(op.strides())));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::PadOp &op, const ExecutionOptions &opts) {
  const auto &operand = lookupValue(sscope, op.operand(), opts);
  const size_t operand_rank = operand.shape().size();
  const auto &padding_value = lookupValue(sscope, op.padding_value(), opts);
  YACL_ENFORCE(padding_value.shape().empty());

  auto edge_padding_low = convertDenseIntElementAttr(op.edge_padding_low());
  YACL_ENFORCE(edge_padding_low.size() == operand_rank);
  auto edge_padding_high = convertDenseIntElementAttr(op.edge_padding_high());
  YACL_ENFORCE(edge_padding_high.size() == operand_rank);
  auto interior_padding = convertDenseIntElementAttr(op.interior_padding());
  YACL_ENFORCE(interior_padding.size() == operand_rank);
  YACL_ENFORCE(std::all_of(interior_padding.begin(), interior_padding.end(),
                           [](int64_t i) { return i >= 0; }));

  sscope->addValue(op.getResult(),
                   kernel::hlo::Pad(hctx, operand, padding_value,
                                    edge_padding_low, edge_padding_high,
                                    interior_padding));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ReverseOp &op, const ExecutionOptions &opts) {
  sscope->addValue(
      op.getResult(),
      kernel::hlo::Reverse(hctx, lookupValue(sscope, op.getOperand(), opts),
                           convertDenseIntElementAttr(op.dimensions())));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ReduceOp &op, const ExecutionOptions &opts) {
  int64_t num_args = op->getNumOperands() / 2;
  std::vector<int64_t> dimensions_to_reduce =
      convertDenseIntElementAttr(op.dimensions());

  std::vector<spu::Value> input_args(num_args);
  std::vector<spu::Value> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = lookupValue(sscope, op.inputs()[i], opts);
    init_values[i] = lookupValue(sscope, op.init_values()[i], opts);
  }

  std::vector<spu::Value> ret = kernel::hlo::Reduce(
      hctx, input_args, init_values, dimensions_to_reduce,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        std::vector<spu::Value> operands;
        operands.reserve(lhs.size() + rhs.size());
        operands.insert(operands.end(), lhs.begin(), lhs.end());
        operands.insert(operands.end(), rhs.begin(), rhs.end());
        return runRegion(executor, hctx, sscope, op.body(), operands);
      });

  const auto &output_shape =
      op->getResultTypes()[0].dyn_cast<mlir::RankedTensorType>().getShape();
  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    sscope->addValue(op->getResult(idx),
                     kernel::hlo::Reshape(hctx, ret[idx], output_shape));
  }
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ReduceWindowOp &op, const ExecutionOptions &opts) {
  int64_t num_args = op->getNumOperands() / 2;

  std::vector<spu::Value> input_args(num_args);
  std::vector<spu::Value> init_values(num_args);

  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = lookupValue(sscope, op.inputs()[i], opts);
    init_values[i] = lookupValue(sscope, op.init_values()[i], opts);
  }

  auto ret_shape = op->getResults()[0]
                       .getType()
                       .dyn_cast<mlir::RankedTensorType>()
                       .getShape();
  auto window_shape = convertDenseIntElementAttr(op.window_dimensions());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().has_value()) {
    window_strides = convertDenseIntElementAttr(*op.window_strides());
  }

  // window dilation
  std::vector<int64_t> window_dilations(window_shape.size(), 1);
  if (op.window_dilations().has_value()) {
    window_dilations = convertDenseIntElementAttr(*op.window_dilations());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().has_value()) {
    const auto v = *op.padding();

    YACL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  // base dilation
  std::vector<int64_t> base_dilation(window_shape.size(), 1);
  if (op.base_dilations().has_value()) {
    base_dilation = convertDenseIntElementAttr(*op.base_dilations());
  }

  kernel::hlo::ReduceWindowConfig config;
  config.window_shape = window_shape;
  config.window_strides = window_strides;
  config.window_dilations = window_dilations;
  config.window_padding = window_padding;
  config.base_dilations = base_dilation;

  auto rets = kernel::hlo::ReduceWindow(
      hctx, input_args, init_values, ret_shape, config,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        std::vector<spu::Value> operands;
        operands.reserve(lhs.size() + rhs.size());
        operands.insert(operands.end(), lhs.begin(), lhs.end());
        operands.insert(operands.end(), rhs.begin(), rhs.end());
        return runRegion(executor, hctx, sscope, op.body(), operands);
      });

  for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
    sscope->addValue(op->getResults()[idx], std::move(rets[idx]));
  }
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ArgMaxOp &op, const ExecutionOptions &opts) {
  auto window_shape = convertDenseIntElementAttr(op.window_dimensions());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().has_value()) {
    window_strides = convertDenseIntElementAttr(*op.window_strides());
  }

  // window dilation
  std::vector<int64_t> window_dilations(window_shape.size(), 1);
  if (op.window_dilations().has_value()) {
    window_dilations = convertDenseIntElementAttr(*op.window_dilations());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().has_value()) {
    const auto v = *op.padding();

    YACL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  // base dilation
  std::vector<int64_t> base_dilation(window_shape.size(), 1);
  if (op.base_dilations().has_value()) {
    base_dilation = convertDenseIntElementAttr(*op.base_dilations());
  }

  auto ret_shape = op->getResults()[0]
                       .getType()
                       .dyn_cast<mlir::RankedTensorType>()
                       .getShape();

  kernel::hlo::ReduceWindowConfig config;
  config.window_shape = window_shape;
  config.window_strides = window_strides;
  config.window_dilations = window_dilations;
  config.window_padding = window_padding;
  config.base_dilations = base_dilation;

  auto ret = kernel::hlo::ArgMax(hctx, lookupValue(sscope, op.input(), opts),
                                 ret_shape, config);

  sscope->addValue(op.getResult(0), ret.first);

  sscope->addValue(op.getResult(1), ret.second);
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::SelectOp &op, const ExecutionOptions &opts) {
  auto pred = lookupValue(sscope, op.pred(), opts);

  auto on_true = lookupValue(sscope, op.on_true(), opts);
  auto on_false = lookupValue(sscope, op.on_false(), opts);

  sscope->addValue(op.getResult(),
                   kernel::hlo::Select(hctx, pred, on_true, on_false));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::RngOp &op, const ExecutionOptions &opts) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  sscope->addValue(
      op.getResult(),
      kernel::hlo::Uniform_rand(hctx, lookupValue(sscope, op.a(), opts),
                                lookupValue(sscope, op.b(), opts), to_shape));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ConvertOp &op, const ExecutionOptions &opts) {
  mlir::pphlo::TypeTools tool;
  auto dst_dtype = getDtypeFromMlirType(op.getType());
  auto dst_vtype = tool.isMPCType<mlir::pphlo::PublicType>(op.getType())
                       ? VIS_PUBLIC
                       : VIS_SECRET;
  auto in = lookupValue(sscope, op.getOperand(), opts);

  sscope->addValue(op.getResult(),
                   kernel::hlo::Cast(hctx, in, dst_vtype, dst_dtype));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::PreferAOp &op, const ExecutionOptions &opts) {
  auto in = lookupValue(sscope, op.operand(), opts);
  auto k0 = kernel::hlo::Cast(hctx, kernel::hlo::Constant(hctx, 0, in.shape()),
                              VIS_PUBLIC, in.dtype());
  sscope->addValue(op.getResult(), kernel::hlo::Add(hctx, in, k0));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::SignOp &op, const ExecutionOptions &opts) {
  auto in = lookupValue(sscope, op.operand(), opts);
  sscope->addValue(op.getResult(), kernel::hlo::Sign(hctx, in));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::BitcastConvertOp &op, const ExecutionOptions &opts) {
  const auto &in_type =
      op.getOperand().getType().dyn_cast<mlir::RankedTensorType>();
  const auto &out_type =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();

  // bitcast should not change total #bytes, so if sizeof(in_t) !=
  // sizeof(out_t) will result to a shape change, thus it's enough to just
  // ensure in_shape == out_shape
  YACL_ENFORCE(in_type.getShape() == out_type.getShape(),
               "bitcast with different size is not supported yet");

  sscope->addValue(
      op.getResult(),
      kernel::hlo::Bitcast(hctx, lookupValue(sscope, op.getOperand(), opts),
                           getDtypeFromMlirType(out_type)));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ConstantOp &op, const ExecutionOptions &opts) {
  const auto &val = op.value();
  const auto &dea = val.dyn_cast<mlir::DenseElementsAttr>();
  const auto &type = val.getType().dyn_cast<mlir::RankedTensorType>();
  const auto &dst_shape = type.getShape();
  const auto &pt_type = getPtTypeFromMlirType(type.getElementType());

  PtBufferView view(dea.getRawData().data(), pt_type,
                    dea.isSplat() ? llvm::ArrayRef<int64_t>() : dst_shape,
                    dea.isSplat() ? std::vector<int64_t>()
                                  : makeCompactStrides(dst_shape));

  sscope->addValue(op.getResult(),
                   kernel::hlo::Constant(hctx, view, dst_shape));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::ClampOp &op, const ExecutionOptions &opts) {
  sscope->addValue(op.getResult(),
                   kernel::hlo::Clamp(hctx,
                                      lookupValue(sscope, op.operand(), opts),
                                      lookupValue(sscope, op.min(), opts),
                                      lookupValue(sscope, op.max(), opts)));
}

void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,
             mlir::pphlo::DbgPrintOp &op, const ExecutionOptions &opts) {
  kernel::hal::dbg_print(hctx, lookupValue(sscope, op.operand(), opts));
}

#define DEFINE_UNIMPLEMENTED_OP(OpName)                                        \
  void execute(OpExecutor *executor, HalContext *hctx, SymbolScope *sscope,    \
               mlir::pphlo::OpName &, const ExecutionOptions &opts) {          \
    YACL_THROW("Lowered op should not occur at backend");                      \
  }

DEFINE_UNIMPLEMENTED_OP(ReturnOp)

#undef DEFINE_UNIMPLEMENTED_OP

} // namespace

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
#include "spu/dialect/pphlo_ops.cc.inc"
      >(op);
}

template <typename OpT, typename... MoreOpT>
static void dispatchOp(OpExecutor *executor, HalContext *hctx,
                       SymbolScope *sscope, mlir::Operation &op,
                       const ExecutionOptions &opts) {
  if (auto casted = llvm::dyn_cast<OpT>(op)) {
    // Execute op
    {
      const auto fn_name = op.getName().getStringRef().str();
      SPU_TRACE_ACTION(GET_CTX_NAME(hctx_), (TR_HLO | TR_LAR), ~TR_HLO,
                       fn_name);
      execute(executor, hctx, sscope, casted, opts);
    }

    // currently we only support config verifier statically.
    constexpr bool kEnableXlaVerifier = false;
    if (kEnableXlaVerifier) {
      XlaVerifier verifier(hctx);
      // handle mixed (int, fxp) multiplication
      if constexpr (std::is_same_v<OpT, mlir::pphlo::MulOp> or
                    std::is_same_v<OpT, mlir::pphlo::DotOp> or
                    std::is_same_v<OpT, mlir::pphlo::DotGeneralOp>) {
        spu::Value lhs = sscope->lookupValue(casted.lhs());
        spu::Value rhs = sscope->lookupValue(casted.rhs());
        spu::Value ret = sscope->lookupValue(casted.getResult());
        mlir::pphlo::TypeTools type_tool;
        auto lhs_type = type_tool.getExpressedType(casted.lhs().getType());
        auto rhs_type = type_tool.getExpressedType(casted.rhs().getType());
        auto ret_type =
            type_tool.getExpressedType(casted.getResult().getType());

        if (lhs_type != ret_type) {
          lhs = kernel::hlo::Cast(hctx, lhs, lhs.vtype(), ret.dtype());
        }
        if (rhs_type != ret_type) {
          rhs = kernel::hlo::Cast(hctx, rhs, rhs.vtype(), ret.dtype());
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
      YACL_THROW("Unhandled mlir op {} at {}", mlirObjectToString(op),
                 mlirObjectToString(op.getLoc()));
    } else {
      dispatchOp<MoreOpT...>(executor, hctx, sscope, op, opts);
    }
  }
}

void PPHloExecutor::runKernelImpl(HalContext *hctx, SymbolScope *sscope,
                                  mlir::Operation &op,
                                  const ExecutionOptions &opts) {
  if (opts.do_log_execution) {
    SPDLOG_INFO("PPHLO {}", mlirObjectToString(op));
  }
  dispatchOp<
#define GET_OP_LIST
#include "spu/dialect/pphlo_ops.cc.inc"
      >(this, hctx, sscope, op, opts);
}

void PPHloExecutor::checkType(mlir::Type mlir_type, const spu::Value &v) const {
}

} // namespace spu::device::pphlo
