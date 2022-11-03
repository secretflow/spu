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

#pragma once

#include "spu/kernel/context.h"
#include "spu/kernel/hlo/casting.h"
#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

template <typename TrueBranch, typename FalseBranch>
std::vector<spu::Value> IfElse(HalContext *ctx, spu::Value condition,
                               const TrueBranch &on_true,
                               const FalseBranch &on_false) {
  if (condition.isSecret() && ctx->rt_config().reveal_secret_condition()) {
    SPDLOG_WARN("Reveal condition variable of If");
    condition = Reveal(ctx, condition);
  }
  bool v = getConditionValue(ctx, condition);

  return (v ? on_true() : on_false());
}

/// While evalation order:
/// 1. Forward all args into cond block
/// 2. Evaluate condition
/// 3. If true -> run body with all args forward into body block
/// 4. If false -> done, set output
template <typename Condition, typename Body>
std::vector<spu::Value> While(HalContext *ctx,
                              absl::Span<const spu::Value> inputs,
                              const Condition &cond, const Body &body) {
  bool warned = false;

  std::vector<spu::Value> ret(inputs.begin(), inputs.end());
  // Push frame
  auto eval_cond = [&](absl::Span<const spu::Value> inputs) -> bool {
    spu::Value c = cond(inputs);

    if (c.isSecret() && ctx->rt_config().reveal_secret_condition()) {
      c = Reveal(ctx, c);
      if (!warned) {
        SPDLOG_WARN("Reveal condition region result of While");
        warned = true;
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
