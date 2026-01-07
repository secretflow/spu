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

#include "libspu/kernel/hal/soprf.h"

#include <vector>

#include "libspu/core/trace.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {

Value soprf(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  if (x.numel() == 0) {
    return x;
  }

  // currently, wo only support LowMC block cipher
  SPU_ENFORCE(ctx->hasKernel("lowmc_b"));
  auto inp = x;

  if (x.isPublic()) {
    inp = _p2s(ctx, x);
  } else if (x.isPrivate()) {
    inp = _v2s(ctx, x);
  }

  auto ret = dynDispatch<spu::Value>(ctx, "lowmc_b", _prefer_b(ctx, inp));

  return ret.setDtype(x.dtype());
}

namespace {
spu::Value _2s(SPUContext* ctx, const Value& x) {
  if (x.isPublic()) {
    return _p2s(ctx, x);
  } else if (x.isPrivate()) {
    return _v2s(ctx, x);
  }
  return x;
}
}  // namespace

Value soprf(SPUContext* ctx, absl::Span<const spu::Value> inputs) {
  SPU_TRACE_HAL_LEAF(ctx, inputs.size());
  // currently, wo only support LowMC block cipher
  SPU_ENFORCE(ctx->hasKernel("multi_key_lowmc_b"));
  SPU_ENFORCE(!inputs.empty(), "inputs should not be empty");
  SPU_ENFORCE(std::all_of(inputs.begin() + 1, inputs.end(),
                          [&inputs](const spu::Value& v) {
                            return v.shape() == inputs.front().shape();
                          }),
              "shape mismatch");
  SPU_ENFORCE(std::all_of(inputs.begin() + 1, inputs.end(),
                          [&inputs](const Value& v) {
                            return v.dtype() == inputs.front().dtype();
                          }),
              "not all element has same dtype");

  if (inputs.front().numel() == 0) {
    return inputs.front();
  }

  std::vector<Value> inp;
  inp.reserve(inputs.size());
  for (const auto& v : inputs) {
    inp.push_back(_prefer_b(ctx, _2s(ctx, v)));
  }

  auto ret = dynDispatch<spu::Value>(ctx, "multi_key_lowmc_b", inp);

  return ret.setDtype(inputs.front().dtype());
}

}  // namespace spu::kernel::hal
