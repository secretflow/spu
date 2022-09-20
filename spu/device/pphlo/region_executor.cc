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

#include "spu/device/pphlo/region_executor.h"

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"

#include "spu/device/frame.h"
#include "spu/device/pphlo/kernels/basic_binary.h"
#include "spu/device/pphlo/kernels/basic_ternary.h"
#include "spu/device/pphlo/kernels/basic_unary.h"
#include "spu/device/pphlo/kernels/casting.h"
#include "spu/device/pphlo/kernels/const.h"
#include "spu/device/pphlo/kernels/control_flow.h"
#include "spu/device/pphlo/kernels/convolution.h"
#include "spu/device/pphlo/kernels/dynamic_slice.h"
#include "spu/device/pphlo/kernels/gather.h"
#include "spu/device/pphlo/kernels/geometrical.h"
#include "spu/device/pphlo/kernels/rand.h"
#include "spu/device/pphlo/kernels/reduce.h"
#include "spu/device/pphlo/kernels/select_and_scatter.h"
#include "spu/device/pphlo/kernels/shift.h"
#include "spu/device/pphlo/kernels/sort.h"
#include "spu/dialect/pphlo_ops.h"

namespace {

std::vector<int64_t>
convertDenseIntElementAttr(const mlir::DenseIntElementsAttr &attr) {
  std::vector<int64_t> ret;

  for (const auto &v : attr.getValues<int64_t>()) {
    ret.emplace_back(v);
  }

  return ret;
}

std::string printLocation(const mlir::Location &loc) {
  std::string pstr;
  llvm::raw_string_ostream ss(pstr);
  loc->print(ss);
  ss.flush();
  return pstr;
}

spu::PtType getPtType(const mlir::Type &type) {
  if (auto ft = type.dyn_cast<mlir::FloatType>()) {
    switch (ft.getWidth()) {
    case 32:
      return spu::PT_F32;
    case 64:
      return spu::PT_F64;
    }
  }
  if (auto it = type.dyn_cast<mlir::IntegerType>()) {
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
  YASL_THROW("Hit unknown pt_type");
}

spu::DataType getDtypeFromMlirType(::mlir::Type mlir_ty) {
  mlir::pphlo::TypeTools tool;
  if (auto int_ty =
          tool.getExpressedType(mlir_ty).dyn_cast<::mlir::IntegerType>()) {
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
      YASL_THROW("unsupported int type {}");
    }
  }
  auto flp_ty = tool.getExpressedType(mlir_ty).dyn_cast<::mlir::FloatType>();
  YASL_ENFORCE(flp_ty, "invalid type");
  return spu::DT_FXP;
}

} // namespace

namespace spu::device::pphlo {

const hal::Value &RegionExecutor::lookupValue(::mlir::Value v) const {
  const auto *val = frame_->getValue(v);
  if (val == nullptr) {
    // Somehow cannot find this value on stack, print a reasonable error
    // message.
    std::string str;
    llvm::raw_string_ostream debug_s(str);
    v.getDefiningOp()->print(debug_s);
    YASL_ENFORCE(false, "Try to get a non-exist value, defined at {}",
                 debug_s.str());
  }
  return *val;
}

#define LOWERED_OP_IMPL(OpName)                                                \
  void RegionExecutor::execute(mlir::pphlo::OpName &) {                        \
    YASL_THROW("Lowered op should not occur at backend");                      \
  }

LOWERED_OP_IMPL(SqrtOp)
LOWERED_OP_IMPL(ReturnOp)
LOWERED_OP_IMPL(NotEqualOp)
LOWERED_OP_IMPL(LessEqualOp)
LOWERED_OP_IMPL(GreaterEqualOp)

#undef LOWERED_OP_IMPL

#define UNIMPL_OP(OpName)                                                      \
  void RegionExecutor::execute(mlir::pphlo::OpName &op) {                      \
    YASL_THROW("Missing Runtime Impl Op {}", op->getName().getStringRef());    \
  }

#undef UNIMPL_OP

std::vector<hal::Value>
RegionExecutor::executeRegion(mlir::Region &region,
                              absl::Span<const hal::Value> inputs) {
  getFrame()->enterRegion();
  if (suppress_type_check_) {
    getFrame()->setTypeCheker(nullptr);
  }

  YASL_ENFORCE(region.getNumArguments() == inputs.size(),
               "Entrypoint function requires {} arguments, which is more than "
               "actual number of inputs {}",
               region.getRegionNumber(), inputs.size());

  for (const auto &blkarg : region.getArguments()) {
    getFrame()->addValue(blkarg, inputs[blkarg.getArgNumber()]);
  }

  auto ret = executeBlock(region.front());
  getFrame()->leaveRegion();
  if (getContext()->rt_config().enable_type_checker()) {
    getFrame()->setTypeCheker(type_checker_);
  }
  return ret;
}

std::vector<hal::Value> RegionExecutor::executeBlock(mlir::Block &block) {
  for (auto &op : block.without_terminator()) {
    dispatchOp<
#define GET_OP_LIST
#include "spu/dialect/pphlo_ops.cc.inc"
        >(op);
  }

  if (auto *termOp = block.getTerminator()) {
    if (!suppress_pphlo_trace_ && hctx_->rt_config().enable_pphlo_trace()) {
      debug_print(*termOp);
    }
    return executeTerminator(*termOp);
  }

  // No terminator
  return {};
}

void RegionExecutor::debug_print(mlir::Operation &op) {
  if (hctx_->lctx() && hctx_->lctx()->Rank() == 0) {
    std::string buf;
    llvm::raw_string_ostream debug_stream(buf);
    op.print(debug_stream);
    SPDLOG_INFO("PPHLO {}", debug_stream.str());
  }
}

std::vector<hal::Value> RegionExecutor::executeTerminator(mlir::Operation &op) {
  if (llvm::isa<mlir::func::ReturnOp>(op) ||
      llvm::isa<mlir::pphlo::ReturnOp>(op)) {
    std::vector<hal::Value> results;
    results.reserve(op.getNumOperands());
    for (const auto operand : op.getOperands()) {
      results.emplace_back(lookupValue(operand));
    }
    return results;
  }
  llvm_unreachable("Unknown block terminator");
}

#define STANDARD_UNARY_OP_EXEC_IMPL(OpName, KernelName)                        \
  void RegionExecutor::execute(mlir::pphlo::OpName &op) {                      \
    const auto in = lookupValue(op.getOperand());                              \
    auto ret = kernel::KernelName(hctx_, in);                                  \
    getFrame()->addValue(op.getResult(), std::move(ret));                      \
  }

STANDARD_UNARY_OP_EXEC_IMPL(ReciprocalOp, Reciprocal)
STANDARD_UNARY_OP_EXEC_IMPL(NegOp, Neg)
STANDARD_UNARY_OP_EXEC_IMPL(ExpOp, Exp)
STANDARD_UNARY_OP_EXEC_IMPL(LogOp, Log)
STANDARD_UNARY_OP_EXEC_IMPL(Log1pOp, Log1p)
STANDARD_UNARY_OP_EXEC_IMPL(FloorOp, Floor)
STANDARD_UNARY_OP_EXEC_IMPL(CeilOp, Ceil)
STANDARD_UNARY_OP_EXEC_IMPL(AbsOp, Abs)
STANDARD_UNARY_OP_EXEC_IMPL(LogisticOp, Logistic)
STANDARD_UNARY_OP_EXEC_IMPL(TanhOp, Tanh)
STANDARD_UNARY_OP_EXEC_IMPL(NotOp, Not)
STANDARD_UNARY_OP_EXEC_IMPL(RsqrtOp, Rsqrt)

#undef STANDARD_UNARY_OP_EXEC_IMPL

#define STANDARD_BINARY_OP_EXEC_IMPL(OpName, KernelName)                       \
  void RegionExecutor::execute(mlir::pphlo::OpName &op) {                      \
    getFrame()->addValue(op.getResult(),                                       \
                         kernel::KernelName(hctx_, lookupValue(op.lhs()),      \
                                            lookupValue(op.rhs())));           \
  }

STANDARD_BINARY_OP_EXEC_IMPL(AddOp, Add)
STANDARD_BINARY_OP_EXEC_IMPL(EqualOp, Equal);
STANDARD_BINARY_OP_EXEC_IMPL(SubOp, Sub)
STANDARD_BINARY_OP_EXEC_IMPL(LessOp, Less)
STANDARD_BINARY_OP_EXEC_IMPL(GreaterOp, Greater)
STANDARD_BINARY_OP_EXEC_IMPL(MulOp, Mul)
STANDARD_BINARY_OP_EXEC_IMPL(MixedMulOp, Mul)
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

void RegionExecutor::execute(mlir::pphlo::DotOp &op) {
  auto ret = kernel::Dot(hctx_, lookupValue(op.lhs()), lookupValue(op.rhs()));

  const auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::TensorType>().getShape();

  getFrame()->addValue(op.getResult(), kernel::Reshape(hctx_, ret, ret_shape));
}

void RegionExecutor::execute(mlir::pphlo::MixedDotOp &op) {
  auto ret = kernel::Dot(hctx_, lookupValue(op.lhs()), lookupValue(op.rhs()));

  const auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::TensorType>().getShape();

  getFrame()->addValue(op.getResult(), kernel::Reshape(hctx_, ret, ret_shape));
}

void RegionExecutor::execute(mlir::pphlo::ConvOp &op) {
  const auto &dnums = op.dimension_numbers();
  const size_t num_spatial_dims = dnums.getOutputSpatialDimensions().size();
  YASL_ENFORCE(num_spatial_dims == dnums.getInputSpatialDimensions().size());
  YASL_ENFORCE(num_spatial_dims == dnums.getKernelSpatialDimensions().size());

  // const auto lhs_shape =
  //     op.lhs().getType().dyn_cast<mlir::TensorType>().getShape();

  // const auto rhs_shape =
  //     op.rhs().getType().dyn_cast<mlir::TensorType>().getShape();

  const auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::TensorType>().getShape();

  // bool fast_path = true;
  // // fast path restrictions
  // // Restriction 1.
  // fast_path &= (op.feature_group_count() == 1);
  // fast_path &= (op.batch_group_count() == 1);

  auto lhs = lookupValue(op.lhs());
  auto rhs = lookupValue(op.rhs());

  std::vector<int64_t> window_strides(dnums.getInputSpatialDimensions().size(),
                                      1);
  if (op.window_strides().hasValue()) {
    for (const auto &iter :
         llvm::enumerate(op.window_strides()->getValues<int64_t>())) {
      window_strides[iter.index()] = iter.value();
    }
  }
  std::vector<int64_t> padding(2 * dnums.getInputSpatialDimensions().size(), 0);
  if (op.padding().hasValue()) {
    for (const auto &iter :
         llvm::enumerate(op.padding()->getValues<int64_t>())) {
      padding[iter.index()] = iter.value();
    }
  }
  std::vector<int64_t> lhs_dilation(dnums.getInputSpatialDimensions().size(),
                                    1);
  if (op.lhs_dilation().hasValue()) {
    for (const auto &iter :
         llvm::enumerate(op.lhs_dilation()->getValues<int64_t>())) {
      lhs_dilation[iter.index()] = iter.value();
    }
  }
  std::vector<int64_t> rhs_dilation(dnums.getInputSpatialDimensions().size(),
                                    1);
  if (op.rhs_dilation().hasValue()) {
    for (const auto &iter :
         llvm::enumerate(op.rhs_dilation()->getValues<int64_t>())) {
      rhs_dilation[iter.index()] = iter.value();
    }
  }

  kernel::ConvolutionConfig config;
  config.featureGroupCount = op.feature_group_count();
  config.batchGroupCount = op.batch_group_count();
  config.window_strides = window_strides;
  config.padding = padding;
  config.lhs_dilation = lhs_dilation;
  config.rhs_dilation = rhs_dilation;
  config.inputBatchDimension = dnums.getInputBatchDimension();
  config.inputFeatureDimension = dnums.getInputFeatureDimension();
  config.inputSpatialDimensions = dnums.getInputSpatialDimensions();
  config.kernelInputFeatureDimension = dnums.getKernelInputFeatureDimension();
  config.kernelOutputFeatureDimension = dnums.getKernelOutputFeatureDimension();
  config.kernelSpatialDimensions = dnums.getKernelSpatialDimensions();
  config.outputBatchDimension = dnums.getOutputBatchDimension();
  config.outputFeatureDimension = dnums.getOutputFeatureDimension();
  config.outputSpatialDimensions = dnums.getOutputSpatialDimensions();

  hal::Value result;
  // FIXME:(xiaochen) Reenable fast path once fixed conv2d
  // if (lhs_shape.size() == 4 && rhs_shape.size() == 4 && fast_path) {
  //   result = kernel::Convolution2D(hctx_, lhs, rhs, config);
  // } else {
  result = kernel::Convolution(hctx_, lhs, rhs, config, ret_shape);
  // }

  getFrame()->addValue(op.getResult(), std::move(result));
}

void RegionExecutor::execute(mlir::pphlo::DynamicUpdateSliceOp &op) {
  // Basic idea here, get a ref slice and update the whole slice..
  // Start indicies
  std::vector<hal::Value> start_indicies(op.start_indices().size());
  const auto &operand = lookupValue(op.operand());
  const auto &update = lookupValue(op.update());

  for (const auto &idx : llvm::enumerate(op.start_indices())) {
    start_indicies[idx.index()] = lookupValue(idx.value());
  }

  getFrame()->addValue(
      op.getResult(),
      kernel::DynamicUpdateSlice(hctx_, operand, update, start_indicies));
}

void RegionExecutor::execute(mlir::pphlo::DynamicSliceOp &op) {
  // Start indicies
  auto iter = op.slice_sizes().getValues<int64_t>();
  std::vector<int64_t> slice_size{iter.begin(), iter.end()};
  const auto &operand = lookupValue(op.operand());
  std::vector<hal::Value> start_indicies(op.start_indices().size());

  for (const auto &idx : llvm::enumerate(op.start_indices())) {
    start_indicies[idx.index()] = lookupValue(idx.value());
  }

  getFrame()->addValue(
      op.getResult(),
      kernel::DynamicSlice(hctx_, operand, slice_size, start_indicies));
}

void RegionExecutor::execute(mlir::pphlo::GatherOp &op) {
  // If input is empty, short circuit
  auto operand = lookupValue(op.operand());
  auto start_indicies = lookupValue(op.start_indices());
  if (operand.numel() == 0) {
    getFrame()->addValue(op.getResult(), operand);
    return;
  }

  const auto &output_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();

  const auto &dim_numbers = op.dimension_numbers();

  kernel::GatherConfig config;
  auto ss = convertDenseIntElementAttr(op.slice_sizes());
  config.sliceSizes = ss;
  config.indexVectorDim = dim_numbers.getIndexVectorDim();
  config.offsetDims = dim_numbers.getOffsetDims();
  config.collapsedSliceDims = dim_numbers.getCollapsedSliceDims();
  config.startIndexMap = dim_numbers.getStartIndexMap();

  getFrame()->addValue(
      op.getResult(),
      kernel::Gather(hctx_, operand, start_indicies, config, output_shape));
}

void RegionExecutor::execute(mlir::pphlo::SortOp &op) {
  auto sort_dim = op.dimension();
  auto is_stable = op.is_stable();
  std::vector<hal::Value> inputs(op->getNumOperands());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs[idx] = lookupValue(op->getOperand(idx));
  }
  auto ret = kernel::Sort(hctx_, inputs, sort_dim, is_stable,
                          [&](absl::Span<const hal::Value> inputs) {
                            auto ret = executeRegion(op.comparator(), inputs);
                            return ret[0];
                          });

  for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
    getFrame()->addValue(op->getResult(idx), std::move(ret[idx]));
  }
}

void RegionExecutor::execute(mlir::pphlo::SelectAndScatterOp &op) {
  auto operand = lookupValue(op.operand());
  auto source = lookupValue(op.source());
  auto init_val = lookupValue(op.init_value());

  auto window_shape =
      convertDenseIntElementAttr(op.window_dimensions().getValue());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().hasValue()) {
    window_strides = convertDenseIntElementAttr(*op.window_strides());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().hasValue()) {
    const auto v = *op.padding();

    YASL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  std::vector<int64_t> window_dilations(window_shape.size(), 1);
  std::vector<int64_t> base_dilations(operand.shape().size(), 1);

  kernel::SelectAndScatterConfig config;
  config.window_shape = window_shape;
  config.window_strides = window_strides;
  config.window_padding = window_padding;
  config.window_dilations = window_dilations;
  config.base_dilations = base_dilations;

  suppress_pphlo_trace_ = true;
  suppress_type_check_ = true;

  auto ret = kernel::SelectAndScatter(
      hctx_, operand, source, init_val,
      [&](const hal::Value &selected, const hal::Value &current) {
        auto ret = executeRegion(op.select(), {selected, current});
        return ret[0];
      },
      config);

  suppress_pphlo_trace_ = false;
  suppress_type_check_ = false;

  getFrame()->addValue(op.getResult(), std::move(ret));
}

void RegionExecutor::execute(mlir::pphlo::IfOp &op) {
  auto conditional = lookupValue(op.condition());

  auto results = kernel::IfElse(
      hctx_, conditional, //
      [&]() { return executeRegion(op.true_branch(), {}); },
      [&]() { return executeRegion(op.false_branch(), {}); });

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    getFrame()->addValue(ret.value(), results[ret.index()]);
  }
}

void RegionExecutor::execute(mlir::pphlo::WhileOp &op) {
  // First inputs vectors
  std::vector<hal::Value> inputs;
  inputs.reserve(op->getNumOperands());

  // Prepare inputs
  for (const auto operand : op->getOperands()) {
    inputs.emplace_back(lookupValue(operand));
  }

  auto ret = kernel::While(
      hctx_, inputs, //
      [&](absl::Span<const hal::Value> inputs) {
        return executeRegion(op.cond(), inputs)[0];
      },
      [&](absl::Span<const hal::Value> inputs) {
        return executeRegion(op.body(), inputs);
      });

  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    getFrame()->addValue(op->getResult(idx), std::move(ret[idx]));
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
      YASL_THROW("{} not implemented for pt_type={}", #NAME, PT_TYPE);         \
    }                                                                          \
  }()

void RegionExecutor::execute(mlir::pphlo::IotaOp &op) {
  const auto &ret_type =
      op.output().getType().dyn_cast<mlir::RankedTensorType>();
  const size_t numel = ret_type.getShape()[op.iota_dimension()];

  auto ret_el_type = type_tools_.getExpressedType(ret_type);
  auto pt_type = getPtType(ret_el_type);

  hal::Value iota_ret;
  DISPATCH_ALL_NONE_BOOL_PT_TYPES(pt_type, "_", [&] {
    iota_ret = kernel::Iota<ScalarT>(hctx_, numel, VIS_PUBLIC);
  });

  if (ret_type.getShape().size() > 1) {
    // Need a broadcast
    iota_ret = kernel::Broadcast(hctx_, iota_ret, ret_type.getShape(), {});
  }

  getFrame()->addValue(op.output(), std::move(iota_ret));
}

void RegionExecutor::execute(mlir::pphlo::RemOp &op) {
  // FIXME: When hal has a remainder, use that
  auto lhs = lookupValue(op.lhs());
  auto rhs = lookupValue(op.rhs());

  auto ret = kernel::Remainder(hctx_, lhs, rhs);
  getFrame()->addValue(op.getResult(), std::move(ret));
}

void RegionExecutor::execute(mlir::pphlo::TransposeOp &op) {
  getFrame()->addValue(
      op.getResult(),
      kernel::Transpose(hctx_, lookupValue(op.getOperand()),
                        convertDenseIntElementAttr(op.permutation())));
}

void RegionExecutor::execute(mlir::pphlo::BroadcastOp &op) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  getFrame()->addValue(
      op.getResult(),
      kernel::Broadcast(hctx_, lookupValue(op.getOperand()), to_shape,
                        convertDenseIntElementAttr(op.broadcast_dimensions())));
}

void RegionExecutor::execute(mlir::pphlo::ReshapeOp &op) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  getFrame()->addValue(
      op.getResult(),
      kernel::Reshape(hctx_, lookupValue(op.getOperand()), to_shape));
}

void RegionExecutor::execute(mlir::pphlo::ConcatenateOp &op) {
  std::vector<hal::Value> values(op->getNumOperands());

  for (size_t idx = 0; idx < op->getNumOperands(); ++idx) {
    values[idx] = lookupValue(op->getOperand(idx));
  }

  // set result
  getFrame()->addValue(op.getResult(),
                       kernel::Concatenate(hctx_, values, op.dimension()));
}

void RegionExecutor::execute(mlir::pphlo::SliceOp &op) {
  getFrame()->addValue(
      op.getResult(),
      kernel::Slice(hctx_, lookupValue(op.getOperand()),
                    convertDenseIntElementAttr(op.start_indices()),
                    convertDenseIntElementAttr(op.limit_indices()),
                    convertDenseIntElementAttr(op.strides())));
}

void RegionExecutor::execute(mlir::pphlo::PadOp &op) {
  const auto &operand = lookupValue(op.operand());
  const size_t operand_rank = operand.shape().size();
  const auto &padding_value = lookupValue(op.padding_value());
  YASL_ENFORCE(padding_value.shape().empty());

  auto edge_padding_low = convertDenseIntElementAttr(op.edge_padding_low());
  YASL_ENFORCE(edge_padding_low.size() == operand_rank);
  auto edge_padding_high = convertDenseIntElementAttr(op.edge_padding_high());
  YASL_ENFORCE(edge_padding_high.size() == operand_rank);
  auto interior_padding = convertDenseIntElementAttr(op.interior_padding());
  YASL_ENFORCE(interior_padding.size() == operand_rank);
  YASL_ENFORCE(std::all_of(interior_padding.begin(), interior_padding.end(),
                           [](int64_t i) { return i >= 0; }));

  getFrame()->addValue(op.getResult(),
                       kernel::Pad(hctx_, operand, padding_value,
                                   edge_padding_low, edge_padding_high,
                                   interior_padding));
}

void RegionExecutor::execute(mlir::pphlo::ReverseOp &op) {
  getFrame()->addValue(
      op.getResult(),
      kernel::Reverse(hctx_, lookupValue(op.getOperand()),
                      convertDenseIntElementAttr(op.dimensions())));
}

void RegionExecutor::errorUnknownOp(mlir::Operation &op) {
  // These lines of code in theory should not hit.
  // If hit, make a proper error message.
  std::string err_str;
  llvm::raw_string_ostream err(err_str);
  op.print(err);
  YASL_THROW("Unhandled mlir op {} at {}", err.str(),
             printLocation(op.getLoc()));
}

void RegionExecutor::execute(mlir::pphlo::ReduceOp &op) {
  int64_t num_args = op->getNumOperands() / 2;
  std::vector<int64_t> dimensions_to_reduce =
      convertDenseIntElementAttr(op.dimensions());

  std::vector<hal::Value> input_args(num_args);
  std::vector<hal::Value> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = lookupValue(op.inputs()[i]);
    init_values[i] = lookupValue(op.init_values()[i]);
  }
  const auto &output_shape =
      op->getResultTypes()[0].dyn_cast<mlir::RankedTensorType>().getShape();

  suppress_type_check_ = true;
  suppress_pphlo_trace_ = true;

  std::vector<hal::Value> ret = kernel::TreeReduce(
      hctx_, input_args, init_values, dimensions_to_reduce, output_shape,
      [&](absl::Span<const hal::Value> lhs, absl::Span<const hal::Value> rhs) {
        std::vector<hal::Value> operands;
        operands.reserve(lhs.size() + rhs.size());
        operands.insert(operands.end(), lhs.begin(), lhs.end());
        operands.insert(operands.end(), rhs.begin(), rhs.end());
        return executeRegion(op.body(), operands);
      });

  suppress_type_check_ = false;
  suppress_pphlo_trace_ = false;

  for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
    getFrame()->addValue(op->getResult(idx),
                         kernel::Reshape(hctx_, ret[idx], output_shape));
  }
}

void RegionExecutor::execute(mlir::pphlo::ReduceWindowOp &op) {
  YASL_ENFORCE(op->getNumResults() == 1,
               "Variadic reduce window is not supported yet");

  const auto &input = lookupValue(op.inputs());
  const auto &init_val = lookupValue(op.init_values());

  auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  auto window_shape = convertDenseIntElementAttr(op.window_dimensions());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().hasValue()) {
    window_strides = convertDenseIntElementAttr(*op.window_strides());
  }

  // window dilation
  std::vector<int64_t> window_dilations(window_shape.size(), 1);
  if (op.window_dilations().hasValue()) {
    window_dilations = convertDenseIntElementAttr(*op.window_dilations());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().hasValue()) {
    const auto v = *op.padding();

    YASL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  // base dilation
  std::vector<int64_t> base_dilation(window_shape.size(), 1);
  if (op.base_dilations().hasValue()) {
    base_dilation = convertDenseIntElementAttr(*op.base_dilations());
  }

  kernel::ReduceWindowConfig config;
  config.window_shape = window_shape;
  config.window_strides = window_strides;
  config.window_dilations = window_dilations;
  config.window_padding = window_padding;
  config.base_dilations = base_dilation;

  suppress_type_check_ = true;
  suppress_pphlo_trace_ = true;
  auto ret =
      kernel::ReduceWindow(hctx_, input, init_val, ret_shape, config,
                           [&](const hal::Value &lhs, const hal::Value &rhs) {
                             return executeRegion(op.body(), {lhs, rhs})[0];
                           });
  suppress_type_check_ = false;
  suppress_pphlo_trace_ = false;

  getFrame()->addValue(op.getResult(), std::move(ret));
}

void RegionExecutor::execute(mlir::pphlo::SelectOp &op) {
  auto pred = lookupValue(op.pred());
  auto on_true = lookupValue(op.on_true());
  auto on_false = lookupValue(op.on_false());

  getFrame()->addValue(op.getResult(),
                       kernel::Select(hctx_, pred, on_true, on_false));
}

void RegionExecutor::execute(mlir::pphlo::RngUniformOp &op) {
  auto to_shape = op.getType().dyn_cast<mlir::RankedTensorType>().getShape();
  getFrame()->addValue(op.getResult(),
                       kernel::Uniform_rand(hctx_, lookupValue(op.a()),
                                            lookupValue(op.b()), to_shape));
}

void RegionExecutor::execute(mlir::pphlo::ConvertOp &op) {
  mlir::pphlo::TypeTools tool;
  auto dst_dtype = getDtypeFromMlirType(op.getType());
  auto dst_vtype = tool.isMPCType<mlir::pphlo::PublicType>(op.getType())
                       ? VIS_PUBLIC
                       : VIS_SECRET;
  auto in = lookupValue(op.getOperand());

  getFrame()->addValue(op.getResult(),
                       kernel::Cast(hctx_, in, dst_vtype, dst_dtype));
}

void RegionExecutor::execute(mlir::pphlo::BitcastConvertOp &op) {
  const auto &in_type =
      op.getOperand().getType().dyn_cast<mlir::RankedTensorType>();
  const auto &out_type =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();

  // bitcast should not change total #bytes, so if sizeof(in_t) !=
  // sizeof(out_t) will result to a shape change, thus it's enough to just
  // ensure in_shape == out_shape
  YASL_ENFORCE(in_type.getShape() == out_type.getShape(),
               "bitcast with different size is not supported yet");

  getFrame()->addValue(op.getResult(),
                       kernel::Bitcast(hctx_, lookupValue(op.getOperand()),
                                       getDtypeFromMlirType(out_type),
                                       op.elsize()));
}

void RegionExecutor::execute(mlir::pphlo::ConstOp &op) {
  const auto &val = op.value();
  const auto &dea = val.dyn_cast<mlir::DenseElementsAttr>();
  const auto &type = val.getType().dyn_cast<mlir::RankedTensorType>();
  const auto &dst_shape = type.getShape();
  const auto &pt_type = getPtType(type.getElementType());

  PtBufferView view(dea.getRawData().data(), pt_type,
                    dea.isSplat() ? llvm::ArrayRef<int64_t>() : dst_shape,
                    dea.isSplat() ? std::vector<int64_t>()
                                  : makeCompactStrides(dst_shape));

  getFrame()->addValue(op.getResult(),
                       kernel::Constant(hctx_, view, dst_shape));
}

void RegionExecutor::execute(mlir::pphlo::ClampOp &op) {
  getFrame()->addValue(op.getResult(),
                       kernel::Clamp(hctx_, lookupValue(op.operand()),
                                     lookupValue(op.min()),
                                     lookupValue(op.max())));
}

void RegionExecutor::execute(mlir::pphlo::DbgPrintOp &op) {
  hal::dbg_print(hctx_, lookupValue(op.operand()));
}

} // namespace spu::device::pphlo
