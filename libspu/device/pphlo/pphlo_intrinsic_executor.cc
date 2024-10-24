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

#include <future>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "spdlog/spdlog.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"

namespace spu::device::pphlo {

namespace {

Value sparse_dot(SPUContext* ctx, const Value& lhs, const Value& rhs,
                 const xt::xarray<bool>& mask, int64_t k_step, int64_t n_step) {
  int64_t m = lhs.shape()[0];
  int64_t k = lhs.shape()[1];
  int64_t n = rhs.shape()[1];

  int64_t k_blocks = (k + k_step - 1) / k_step;
  int64_t n_blocks = (n + n_step - 1) / n_step;

  std::vector<spu::Value> results(n_blocks);

  auto get_mask = [&](int64_t m_start, int64_t m_end, int64_t k_start,
                      int64_t k_end) -> bool {
    auto mask_slice =
        xt::view(mask, xt::range(m_start, m_end), xt::range(k_start, k_end));

    xt::xarray<bool> max = xt::amax(mask_slice);

    return max.front();
  };

  for (int64_t n_idx = 0; n_idx < n_blocks; n_idx++) {
    auto n_len = std::min(n_step, n - n_idx * n_step);
    results[n_idx] = kernel::hlo::Constant(ctx, 0.0, {m, n_len});
  }

  for (int64_t k_idx = 0; k_idx < k_blocks; k_idx++) {
    auto k_len = std::min(k_step, k - k_idx * k_step);
    Index lhs_slice_begin({0, k_idx * k_step});
    Index lhs_slice_end({m, k_idx * k_step + k_len});

    auto lhs_slice =
        kernel::hal::slice(ctx, lhs, lhs_slice_begin, lhs_slice_end);

    for (int64_t n_idx = 0; n_idx < n_blocks; n_idx++) {
      auto n_len = std::min(n_step, n - n_idx * n_step);
      Index rhs_slice_begin({k_idx * k_step, n_idx * n_step});
      Index rhs_slice_end({k_idx * k_step + k_len, n_idx * n_step + n_len});

      if (!get_mask(rhs_slice_begin[0], rhs_slice_end[0], rhs_slice_begin[1],
                    rhs_slice_end[1])) {
        continue;
      }

      auto rhs_slice =
          kernel::hal::slice(ctx, rhs, rhs_slice_begin, rhs_slice_end);

      auto slice_dot = kernel::hal::matmul(ctx, lhs_slice, rhs_slice);

      results[n_idx] = kernel::hal::add(ctx, results[n_idx], slice_dot);
    }
  }

  return kernel::hal::concatenate(ctx, results, 1);
}

spu::Value sparse_dot_general(SPUContext* ctx, const Value& lhs,
                              const Value& rhs, const Value& mask,
                              int64_t k_step, int64_t n_step) {
  const bool fork_able = ctx->prot()->hasLowCostFork();
  const bool has_rm_mmul = ctx->prot()->hasKernel("rm_mmul_aa");

  int64_t num_batch = lhs.shape()[0];

  Index lhs_slice_begin(3, 0);
  Index lhs_slice_end(lhs.shape().begin(), lhs.shape().end());
  Index rhs_slice_begin(3, 0);
  Index rhs_slice_end(rhs.shape().begin(), rhs.shape().end());
  Strides strides(lhs.shape().size(), 1);

  Shape lhs_slice_shape{lhs.shape()[1], lhs.shape()[2]};
  Shape rhs_slice_shape{rhs.shape()[1], rhs.shape()[2]};
  Shape ret_slice_shape{1, lhs.shape()[1], rhs.shape()[2]};

  auto thread = [&](std::unique_ptr<SPUContext> ctx, const Value& lhs,
                    const Value& rhs, const Value& mask) {
    return sparse_dot(ctx.get(), lhs, rhs,
                      kernel::hal::dump_public_as<bool>(ctx.get(), mask),
                      k_step, n_step);
  };

  std::vector<std::future<Value>> futures(num_batch);
  std::vector<spu::Value> results;
  results.reserve(num_batch);

  for (int64_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    lhs_slice_begin[0] = batch_idx;
    lhs_slice_end[0] = batch_idx + 1;
    rhs_slice_begin[0] = batch_idx;
    rhs_slice_end[0] = batch_idx + 1;
    auto lhs_slice = kernel::hal::reshape(
        ctx,
        kernel::hal::slice(ctx, lhs, lhs_slice_begin, lhs_slice_end, strides),
        lhs_slice_shape);
    auto mask_slice = kernel::hal::reshape(
        ctx,
        kernel::hal::slice(ctx, mask, rhs_slice_begin, rhs_slice_end, strides),
        rhs_slice_shape);
    auto rhs_slice = kernel::hal::reshape(
        ctx,
        kernel::hal::slice(ctx, rhs, rhs_slice_begin, rhs_slice_end, strides),
        rhs_slice_shape);

    if (has_rm_mmul) {
      SPDLOG_WARN("has_rm_mmul");
      // cheetah test
      results.push_back(kernel::hal::reshape(
          ctx, kernel::hal::rand_mask_matmul(ctx, lhs_slice, rhs_slice),
          ret_slice_shape));
    } else {
      if (fork_able) {
        futures[batch_idx] = std::async(std::launch::async, thread, ctx->fork(),
                                        lhs_slice, rhs_slice, mask_slice);
      } else {
        results.push_back(kernel::hal::reshape(
            ctx,
            sparse_dot(ctx, lhs_slice, rhs_slice,
                       kernel::hal::dump_public_as<bool>(ctx, mask_slice),
                       k_step, n_step),
            ret_slice_shape));
      }
    }
  }

  if (!has_rm_mmul && fork_able) {
    for (auto& f : futures) {
      auto r = f.get();
      results.push_back(kernel::hal::reshape(ctx, r, ret_slice_shape));
    }
  }

  return kernel::hal::concatenate(ctx, results, 0);
}

}  // namespace

std::vector<Value> intrinsic_dispatcher(SPUContext* ctx, llvm::StringRef name,
                                        absl::Span<const Value> inputs) {
  if (name == "sparse_dot_general") {
    auto lhs = inputs[0];
    auto rhs = inputs[1];
    auto mask = inputs[2];
    auto prune_pattern = inputs[3];
    const bool has_rm_mmul = ctx->prot()->hasKernel("rm_mmul_aa");
    SPDLOG_INFO("Executing {} mask = {}, pattern = {}", name.str(), mask,
                prune_pattern);

    if (!has_rm_mmul) {
      if (mask.isSecret()) {
        mask = kernel::hal::reveal(ctx, mask);
      }
      if (prune_pattern.isSecret()) {
        prune_pattern = kernel::hal::reveal(ctx, prune_pattern);
      }
      SPU_ENFORCE(prune_pattern.numel() == 2);
    }

    // Add leading batch dim if missing
    if (lhs.shape().ndim() == 2) {
      lhs = kernel::hlo::Reshape(ctx, lhs, {1, lhs.shape()[0], lhs.shape()[1]});
    }

    if (mask.shape().ndim() == 2) {
      mask = kernel::hlo::Reshape(ctx, mask,
                                  {1, mask.shape()[0], mask.shape()[1]});
    }

    if (rhs.shape().ndim() == 2) {
      rhs = kernel::hlo::Reshape(ctx, rhs, {1, rhs.shape()[0], rhs.shape()[1]});
    }

    SPU_ENFORCE(mask.shape() == rhs.shape());
    if (!has_rm_mmul) {
      auto prune_steps =
          kernel::hal::dump_public_as<int64_t>(ctx, prune_pattern);
      prune_steps.reshape({2});
      return {sparse_dot_general(ctx, lhs, rhs, mask, prune_steps(0),
                                 prune_steps(1))};
    } else {
      return {sparse_dot_general(ctx, lhs, rhs, mask, 0, 0)};
    }
  }
  // DO-NOT-EDIT: Add_DISPATCH_CODE

  // Default: Identity function
  if (name == "example") {
    SPDLOG_INFO("Calling example intrinsic");
    return {inputs.begin(), inputs.end()};
  }

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

  SPU_THROW("Unhandled intrinsic call {}", name.str());
}

}  // namespace spu::device::pphlo
