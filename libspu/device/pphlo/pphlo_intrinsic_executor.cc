// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/device/pphlo/pphlo_intrinsic_executor.h"

#include "spdlog/spdlog.h"

#include "libspu/device/intrinsic_table.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/indexing.h"
#include "libspu/kernel/hlo/rank.h"

namespace spu::device::pphlo {

std::vector<Value> intrinsic_dispatcher(SPUContext* ctx,
                                        mlir::spu::pphlo::CustomCallOp& call,
                                        absl::Span<const Value> inputs) {
  // FIXME: This should be something register by protocol
  auto name = call.getCallTargetName();
  if (name == "example_binary") {
    SPDLOG_INFO("Binary example, input0 = {}, input1 = {}", inputs[0],
                inputs[1]);

    Shape result_shape = {inputs[0].shape()[0] + inputs[1].shape()[0],
                          inputs[0].shape()[1] + inputs[1].shape()[1]};

    auto zeros = kernel::hlo::Constant(ctx, 0, result_shape);

    if (inputs[0].isSecret() || inputs[1].isSecret()) {
      zeros = kernel::hlo::Cast(ctx, zeros, VIS_SECRET, inputs[0].dtype());
    } else {
      zeros = kernel::hlo::Cast(ctx, zeros, VIS_PUBLIC, inputs[0].dtype());
    }

    return {zeros};
  }

  if (name == MAKE_CACHED_VAR) {
    if (ctx->hasKernel("beaver_cache")) {
      SPU_ENFORCE(inputs.size() == 1);
      dynDispatch(ctx, "beaver_cache", inputs[0], true);
    }

    return {inputs[0]};
  }

  if (name == DROP_CACHED_VAR) {
    if (ctx->hasKernel("beaver_cache")) {
      SPU_ENFORCE(inputs.size() > 0);
      dynDispatch(ctx, "beaver_cache", inputs[0], false);
    }

    return {inputs[0]};
  }
  // DO-NOT-EDIT: Add_DISPATCH_CODE

  // Default: Identity function
  if (name == "example") {
    SPDLOG_INFO("Calling example intrinsic");
    return {inputs.begin(), inputs.end()};
  }

  if (name == DBG_PRINT) {
    kernel::hal::dbg_print(ctx, inputs[0]);
    return {};
  }

  if (name == ERF) {
    SPU_ENFORCE(inputs.size() == 1 && inputs[0].isFxp());
    return {kernel::hal::f_erf(ctx, inputs[0])};
  }

  if (name == TOPK) {
    SPU_ENFORCE(inputs.size() == 1);
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("mhlo.attributes"));
    auto k = mlir::dyn_cast<mlir::IntegerAttr>(attr.get("k")).getInt();
    auto largest =
        mlir::dyn_cast<mlir::BoolAttr>(attr.get("largest")).getValue();

    auto value_only = false;

    if (auto value_only_attr = attr.get("value_only")) {
      value_only = mlir::dyn_cast<mlir::BoolAttr>(value_only_attr).getValue();
    }

    if (auto k_hi_attr = attr.get("k_hi")) {
      auto k_hi = mlir::dyn_cast<mlir::IntegerAttr>(k_hi_attr).getInt();
      return kernel::hlo::TopK(ctx, inputs[0], k, k_hi, largest, value_only);
    }

    return kernel::hlo::TopK(ctx, inputs[0], k, -1, largest, value_only);
  }

  if (name == GATHER) {
    kernel::hlo::GatherConfig config;
    const auto& output_shape =
        mlir::dyn_cast<mlir::RankedTensorType>(call.getResults()[0].getType())
            .getShape();
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("pphlo.attributes"));

    config.sliceSizes =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("slice_sizes"))
            .asArrayRef();
    config.indexVectorDim =
        mlir::dyn_cast<mlir::IntegerAttr>(attr.get("index_vector_dim"))
            .getInt();
    config.offsetDims =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("offset_dims"))
            .asArrayRef();
    config.collapsedSliceDims = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
                                    attr.get("collapsed_slice_dims"))
                                    .asArrayRef();
    config.startIndexMap =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("start_index_map"))
            .asArrayRef();

    return {
        kernel::hlo::Gather(ctx, inputs[0], inputs[1], config, output_shape)};
  }

  if (name == PREFER_A) {
    if (ctx->config().protocol() == ProtocolKind::CHEETAH) {
      // NOTE(juhou): For 2PC, MulAB uses COT which is efficient and accurate
      // than MulAA that needs HE. Thus we just by-pass the PreferAOp for 2PC.
      return {inputs[0]};
    }
    auto k0 =
        kernel::hlo::Cast(ctx, kernel::hlo::Constant(ctx, 0, inputs[0].shape()),
                          VIS_PUBLIC, inputs[0].dtype());
    return {kernel::hlo::Add(ctx, inputs[0], k0)};
  }

  SPU_THROW("Unhandled intrinsic call {}", name.str());
}

}  // namespace spu::device::pphlo
