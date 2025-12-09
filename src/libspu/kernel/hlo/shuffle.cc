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

#include "libspu/kernel/hlo/shuffle.h"

#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hlo/sort.h"
#include "libspu/kernel/hal/constants.h"
#include "xtensor/xadapt.hpp"

namespace spu::kernel::hlo {

namespace {

spu::Value _2s(SPUContext* ctx, const Value& x) {
  if (x.isPublic()) {
    return hal::_p2s(ctx, x);
  } else if (x.isPrivate()) {
    return hal::_v2s(ctx, x);
  }
  return x;
}

}  // namespace

std::vector<spu::Value> Shuffle(SPUContext* ctx,
                                absl::Span<const spu::Value> inputs,
                                int64_t axis) {
  SPU_ENFORCE_GT(inputs.size(), 0U);
  auto input_shape = inputs[0].shape();
  SPU_ENFORCE(std::all_of(inputs.begin() + 1, inputs.end(),
                          [&](const spu::Value& v) {
                            return v.shape() == input_shape;
                          }),
              "all inputs should have the same shape");

  // edge case: empty or single element tensor
  if (inputs[0].numel() <= 1) {
    return std::vector<spu::Value>(inputs.begin(), inputs.end());
  }

  // TODO: Rename permute-related kernels
  if (ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am")) {
    auto shuffle_fn = [&](absl::Span<const spu::Value> input) {
      std::vector<spu::Value> rets;
      auto rand_perm = hal::_rand_perm_s(ctx, {input_shape.dim(axis)});
      for (const auto& inp : input) {
        rets.emplace_back(
            hal::_perm_ss(ctx, _2s(ctx, inp), rand_perm).setDtype(inp.dtype()));
      }
      return rets;
    };
    return hal::permute(ctx, inputs, axis, shuffle_fn);
  }

  SPU_ENFORCE_LT(axis, static_cast<int64_t>(input_shape.size()));
  spu::Value rand = hal::random(ctx, VIS_SECRET, DT_U64, input_shape);

  std::vector<spu::Value> inputs_to_sort(inputs.begin(), inputs.end());
  inputs_to_sort.insert(inputs_to_sort.begin(), rand);

  auto outputs =
      SimpleSort(ctx, inputs_to_sort, axis, hal::SortDirection::Ascending);

  return std::vector<spu::Value>(outputs.begin() + 1, outputs.end());
}


std::pair<std::vector<spu::Value>, spu::Value> shuffle_with_perm(
  SPUContext* ctx, absl::Span<const spu::Value> inputs, int64_t axis) {
SPU_ENFORCE_GT(inputs.size(), 0U);
auto input_shape = inputs[0].shape();
SPU_ENFORCE(std::all_of(inputs.begin() + 1, inputs.end(),
                        [&](const spu::Value& v) {
                          return v.shape() == input_shape;
                        }),
            "all inputs should have the same shape");

// edge case: empty or single element tensor
if (inputs[0].numel() <= 1) {
  // 构造一个全0的 dummy perm 返回，保持语义完整
  std::vector<int64_t> zero_idx(input_shape.numel(), 0);
  auto dummy_pi = hal::constant(ctx, xt::adapt(zero_idx), DT_I64, input_shape);
  return {std::vector<spu::Value>(inputs.begin(), inputs.end()), _2s(ctx, dummy_pi)};
}

// ----------------------------------------------------------------
// 路径 1: Permute Protocol (Fast Path)
// ----------------------------------------------------------------
if (ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am")) {
  // 1. 在外面生成 rand_perm (这就是 pi)
  auto rand_perm = hal::_rand_perm_s(ctx, {input_shape.dim(axis)});

  auto shuffle_fn = [&](absl::Span<const spu::Value> input) {
    std::vector<spu::Value> rets;
    // 2. 这里的逻辑没变，只是使用了外部捕获的 rand_perm
    for (const auto& inp : input) {
      rets.emplace_back(
          hal::_perm_ss(ctx, _2s(ctx, inp), rand_perm).setDtype(inp.dtype()));
    }
    return rets;
  };
  
  // 3. 依然使用 hal::permute，不做大改动
  auto results = hal::permute(ctx, inputs, axis, shuffle_fn);
  
  // 4. 返回 {结果, pi}
  return {results, rand_perm};
}

// ----------------------------------------------------------------
// 路径 2: Sort Protocol (Fallback Path)
// ----------------------------------------------------------------
SPU_ENFORCE_LT(axis, static_cast<int64_t>(input_shape.size()));

// 1. 生成随机 Key
spu::Value rand_key = hal::random(ctx, VIS_SECRET, DT_U64, input_shape);

// 2. 【新增】生成索引向量 (Iota)
// 注意：SimpleSort 对整体形状进行排序，这里生成与 input_shape 一致的索引
std::vector<int64_t> indices_vec(input_shape.numel());
std::iota(indices_vec.begin(), indices_vec.end(), 0);

auto indices = hal::constant(ctx, xt::adapt(indices_vec), DT_I64, input_shape);
indices = _2s(ctx, indices); // 转为 Secret

// 3. 组装待排序列表: [Key, Data..., Indices]
std::vector<spu::Value> inputs_to_sort(inputs.begin(), inputs.end());
inputs_to_sort.insert(inputs_to_sort.begin(), rand_key); // 头部加 Key
inputs_to_sort.push_back(indices);                       // 尾部加 Indices

// 4. 排序
auto outputs =
    SimpleSort(ctx, inputs_to_sort, axis, hal::SortDirection::Ascending);

// 5. 拆解结果
// outputs[0]: Key (丢弃)
// outputs[1 ... N]: 打乱后的数据
// outputs[N+1]: 打乱后的 Indices (即 pi)

spu::Value pi = outputs.back();

std::vector<spu::Value> shuffled_inputs;
shuffled_inputs.reserve(inputs.size());
// 提取中间的数据部分
for(size_t i = 1; i < outputs.size() - 1; ++i) {
    shuffled_inputs.push_back(outputs[i]);
}

return {shuffled_inputs, pi};
}

}  // namespace spu::kernel::hlo
