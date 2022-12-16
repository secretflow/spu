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

#include "spu/kernel/hlo/control_flow.h"

#include "spu/kernel/hal/polymorphic.h"
#include "spu/kernel/hal/type_cast.h"
#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

std::vector<spu::Value> IfElse(HalContext *ctx, const spu::Value &condition,
                               const BranchFcnT &on_true,
                               const BranchFcnT &on_false) {
  if (condition.isSecret()) {
    auto true_ret = on_true();
    auto false_ret = on_false();

    YACL_ENFORCE(true_ret.size() == false_ret.size());

    std::vector<spu::Value> selected(true_ret.size());
    for (size_t idx = 0; idx < true_ret.size(); ++idx) {
      selected[idx] =
          hal::select(ctx, condition, true_ret[idx], false_ret[idx]);
    }

    return selected;
  } else {
    bool v = getConditionValue(ctx, condition);

    return (v ? on_true() : on_false());
  }
}

std::vector<spu::Value> While(HalContext *ctx,
                              absl::Span<const spu::Value> inputs,
                              const ConditionFcnT &cond, const BodyFcnT &body) {
  bool warned = false;

  std::vector<spu::Value> ret(inputs.begin(), inputs.end());
  // Push frame
  auto eval_cond = [&](absl::Span<const spu::Value> inputs) -> bool {
    spu::Value c = cond(inputs);

    if (c.isSecret()) {
      if (ctx->rt_config().reveal_secret_condition()) {
        c = hal::reveal(ctx, c);
        if (!warned) {
          SPDLOG_WARN("Reveal condition region result of While");
          warned = true;
        }
      } else {
        YACL_THROW("While with secret condition is not supported");
      }
    }

    return getConditionValue(ctx, c);
  };

  while (eval_cond(ret)) {
    // dispatch body
    ret = body(ret);
  }

  return ret;
}

}  // namespace spu::kernel::hlo