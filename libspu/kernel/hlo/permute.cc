// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/permute.h"

#include "libspu/core/context.h"

namespace spu::kernel::hlo {

namespace {

bool check_permute_kernel(SPUContext* ctx) {
  // TODO: Do checks according to visibility of inputs and perm later.
  return ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am") &&
         ctx->hasKernel("perm_ap") && ctx->hasKernel("inv_perm_am") &&
         ctx->hasKernel("inv_perm_ap");
}
}  // namespace

std::vector<spu::Value> InvPermute(SPUContext* ctx,
                                   absl::Span<const spu::Value> inputs,
                                   const spu::Value& perm, int64_t perm_dim) {
  SPU_ENFORCE(check_permute_kernel(ctx),
              "permute related kernel not supported");

  auto inv_perm_fn = [&](absl::Span<const spu::Value> input) {
    return hal::apply_inv_permute_1d(ctx, input, perm);
  };

  return hal::permute(ctx, inputs, perm_dim, inv_perm_fn);
};

std::vector<spu::Value> Permute(SPUContext* ctx,
                                absl::Span<const spu::Value> inputs,
                                const spu::Value& perm, int64_t perm_dim) {
  SPU_ENFORCE(check_permute_kernel(ctx),
              "permute related kernel not supported");

  auto perm_fn = [&](absl::Span<const spu::Value> input) {
    return hal::apply_permute_1d(ctx, input, perm);
  };

  return hal::permute(ctx, inputs, perm_dim, perm_fn);
}
}  // namespace spu::kernel::hlo
