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

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hlo {

using BranchFcnT = std::function<std::vector<spu::Value>()>;

std::vector<spu::Value> IfElse(SPUContext *ctx, const spu::Value &condition,
                               const BranchFcnT &on_true,
                               const BranchFcnT &on_false);

std::vector<spu::Value> Case(SPUContext *ctx, const spu::Value &index,
                             absl::Span<const BranchFcnT> branches);

/// While evaluation order:
/// 1. Forward all args into cond block
/// 2. Evaluate condition
/// 3. If true -> run body with all args forward into body block
/// 4. If false -> done, set output
using ConditionFcnT = std::function<spu::Value(absl::Span<const spu::Value>)>;
using BodyFcnT =
    std::function<std::vector<spu::Value>(absl::Span<const spu::Value>)>;
std::vector<spu::Value> While(SPUContext *ctx,
                              absl::Span<const spu::Value> inputs,
                              const ConditionFcnT &cond, const BodyFcnT &body);

}  // namespace spu::kernel::hlo
