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

#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
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
  // DO-NOT-EDIT: Add_DISPATCH_CODE

  // Default: Identity function
  if (name == "example") {
    SPDLOG_INFO("Calling example intrinsic");
    return {inputs.begin(), inputs.end()};
  }

  if (name == "dbg_print") {
    kernel::hal::dbg_print(ctx, inputs[0]);
    return {};
  }

  if (name == "mhlo.erf") {
    SPU_ENFORCE(inputs.size() == 1 && inputs[0].isFxp());
    return {kernel::hal::f_erf(ctx, inputs[0])};
  }

  if (name == "mhlo.topk") {
    SPU_ENFORCE(inputs.size() == 1);
    auto attr =
        call->getAttr("mhlo.attributes").dyn_cast<mlir::DictionaryAttr>();
    auto k = attr.get("k").dyn_cast<mlir::IntegerAttr>().getInt();
    auto largest = attr.get("largest").dyn_cast<mlir::BoolAttr>().getValue();

    auto value_only = false;

    if (auto value_only_attr = attr.get("value_only")) {
      value_only = value_only_attr.dyn_cast<mlir::BoolAttr>().getValue();
    }

    if (auto k_hi_attr = attr.get("k_hi")) {
      auto k_hi = k_hi_attr.dyn_cast<mlir::IntegerAttr>().getInt();
      return kernel::hlo::TopK(ctx, inputs[0], k, k_hi, largest, value_only);
    }

    return kernel::hlo::TopK(ctx, inputs[0], k, -1, largest, value_only);
  }

  SPU_THROW("Unhandled intrinsic call {}", name.str());
}

}  // namespace spu::device::pphlo
