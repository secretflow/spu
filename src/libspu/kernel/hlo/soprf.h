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

#include "libspu/kernel/hal/soprf.h"

namespace spu::kernel::hlo {

// shared oblivious PRF
// ret = PRF(x, key), but with x, key in secret share.
// However, for safety, key should be 128 bits long, but `x` may be 64 bits or
// even 32 bits, it's hard to pass another `key` param with FM128, so we just
// generate a shared key inside the kernel.
// TODO: add `key` as a param
Value SoPrf(SPUContext* ctx, const Value& x);

// Multi-Key version of shared oblivious PRF
// We use the scheme in:
// REF: https://eprint.iacr.org/2019/518
//
// Warning: There may exist collision if you feed too many keys, although we
// limit the probability to be less than 2^{-40} in almost situations;
Value SoPrf(SPUContext* ctx, absl::Span<const spu::Value> inputs);

}  // namespace spu::kernel::hlo
