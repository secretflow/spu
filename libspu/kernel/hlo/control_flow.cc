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

#include "libspu/kernel/hlo/control_flow.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/utils.h"

// Allow runtime to reveal `secret variable` use as while
// condition result, debug purpose only.
#define ENABLE_DEBUG_ONLY_REVEAL_SECRET_CONDITION false

namespace spu::kernel::hlo {

std::vector<spu::Value> IfElse(SPUContext *ctx, const spu::Value &condition,
                               const BranchFcnT &on_true,
                               const BranchFcnT &on_false) {
  if (condition.isSecret()) {
    auto true_ret = on_true();
    auto false_ret = on_false();

    SPU_ENFORCE(true_ret.size() == false_ret.size());

    std::vector<spu::Value> selected(true_ret.size());
    for (size_t idx = 0; idx < true_ret.size(); ++idx) {
      selected[idx] =
          hal::select(ctx, condition, true_ret[idx], false_ret[idx]);
    }

    return selected;
  } else {
    bool v = getBooleanValue(ctx, condition);

    return (v ? on_true() : on_false());
  }
}

std::vector<spu::Value> Case(SPUContext *ctx, const spu::Value &index,
                             absl::Span<const BranchFcnT> branches) {
  SPU_ENFORCE(index.isInt());
  if (index.isPublic()) {
    auto idx = getI32Value(ctx, index);
    auto upper_bound = static_cast<int32_t>(branches.size() - 1);
    // If idx < 0 or idx >= N, ran N-1 branch
    idx = (idx < 0 || idx > upper_bound) ? upper_bound : idx;
    return branches[idx]();
  } else {
    // Clamp value first
    auto lower_bound = hal::zeros(ctx, index.dtype());
    auto upper_bound =
        hal::constant(ctx, static_cast<int32_t>(branches.size() - 1), DT_I32);
    auto p = hal::bitwise_or(ctx, hal::less(ctx, index, lower_bound),
                             hal::greater(ctx, index, upper_bound));
    auto normalized_index = hal::select(ctx, p, upper_bound, index);

    // create 0,...,N-1
    auto indices = hlo::Iota(ctx, DT_I32, branches.size());
    // Build mask
    auto masks =
        hal::equal(ctx, indices,
                   hal::broadcast_to(ctx, normalized_index, indices.shape()));

    std::vector<std::vector<spu::Value>> values;
    for (int64_t branch_id = 0;
         branch_id < static_cast<int64_t>(branches.size()); ++branch_id) {
      auto r = branches[branch_id]();
      // Slice mask
      auto mask_i = hal::slice(ctx, masks, {branch_id}, {branch_id + 1}, {});

      for (auto &ret : r) {
        Value mask_i_b;
        if (ret.numel() == mask_i.numel()) {
          mask_i_b = hal::reshape(ctx, mask_i, ret.shape());
        } else {
          mask_i_b = hal::broadcast_to(ctx, mask_i, ret.shape(), {});
        }

        ret = hal::mul(ctx, ret, mask_i_b);
      }
      values.emplace_back(std::move(r));
    }

    // Collect results
    std::vector<spu::Value> results(values.front().size());

    for (int64_t result_id = 0;
         result_id < static_cast<int64_t>(values.front().size()); ++result_id) {
      auto r = values.front()[result_id];
      for (int64_t branch_id = 1;
           branch_id < static_cast<int64_t>(branches.size()); ++branch_id) {
        r = hal::add(ctx, r, values[branch_id][result_id]);
      }
      results[result_id] = r;
    }
    return results;
  }
}

std::vector<spu::Value> While(SPUContext *ctx,
                              absl::Span<const spu::Value> inputs,
                              const ConditionFcnT &cond, const BodyFcnT &body) {
  bool warned = false;

  std::vector<spu::Value> ret(inputs.begin(), inputs.end());
  // Push frame
  auto eval_cond = [&](absl::Span<const spu::Value> inputs) -> bool {
    spu::Value c = cond(inputs);

    if (c.isSecret()) {
      if constexpr (ENABLE_DEBUG_ONLY_REVEAL_SECRET_CONDITION) {
        c = hal::reveal(ctx, c);
        if (!warned) {
          SPDLOG_WARN("Reveal condition region result of While");
          warned = true;
        }
      } else {
        SPU_THROW("While with secret condition is not supported");
      }
    }

    return getBooleanValue(ctx, c);
  };

  while (eval_cond(ret)) {
    // dispatch body
    ret = body(ret);
  }

  return ret;
}

}  // namespace spu::kernel::hlo
