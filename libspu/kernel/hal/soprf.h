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

#pragma once

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

// Shared Oblivious PRF
// ret = PRF(x, key), but with x, key in secret share.
// now, `key` is generated inside kernel to guarantee the 128-bits security.
Value soprf(SPUContext* ctx, const Value& x);

// Multi-Key version of shared oblivious PRF
// We use the scheme in:
// REF: https://eprint.iacr.org/2019/518
//
// Warning: There may exist collision if you feed too many keys, although we
// limit the probability to be less than 2^{-40} in almost situations;
Value soprf(SPUContext* ctx, absl::Span<const spu::Value> inputs);

}  // namespace spu::kernel::hal
