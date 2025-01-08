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
  if (inputs[0].numel() == 0) {
    return std::vector<spu::Value>(inputs.begin(), inputs.end());
  }
  auto input_shape = inputs[0].shape();

  // TODO: Rename permute-related kernels
  if (ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am")) {
    auto shuffle_fn = [&](absl::Span<const spu::Value> input) {
      std::vector<spu::Value> rets;
      auto rand_perm = hal::_rand_perm_s(ctx, input_shape);
      for (size_t i = 0; i < input.size(); ++i) {
        rets.emplace_back(hal::_perm_ss(ctx, _2s(ctx, input[i]), rand_perm)
                              .setDtype(input[i].dtype()));
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

}  // namespace spu::kernel::hlo
