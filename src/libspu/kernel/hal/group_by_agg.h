// Copyright 2025 Ant Group Co., Ltd.
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

#include "absl/types/span.h"

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

// A quick path for groupby sum, when:
//   1. all keys are private and have the same owner
//
// Warning: we skip the sanity checks here, which should be done by the caller
// if the visibility requirements are not met, the performance may be degraded.
std::vector<Value> private_groupby_sum_1d(
    SPUContext *ctx, absl::Span<spu::Value const> keys,
    absl::Span<spu::Value const> payloads);
}  // namespace spu::kernel::hal
