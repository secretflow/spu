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

#include "libspu/kernel/hlo/basic_ternary.h"

#include "libspu/kernel/hal/complex.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hlo {

spu::Value Select(SPUContext *ctx, const spu::Value &pred,
                  const spu::Value &on_true, const spu::Value &on_false) {
  if (on_true.isComplex()) {
    SPU_ENFORCE(on_false.isComplex());
    auto pred_a = hal::_prefer_a(ctx, pred);
    auto r = hal::select(ctx, pred_a, hal::real(ctx, on_true),
                         hal::real(ctx, on_false));
    auto i = hal::select(ctx, pred_a, hal::imag(ctx, on_true),
                         hal::imag(ctx, on_false));
    return hal::complex(ctx, r, i);
  }
  return hal::select(ctx, pred, on_true, on_false);
}

spu::Value Clamp(SPUContext *ctx, const spu::Value &operand,
                 const spu::Value &min, const spu::Value &max) {
  SPU_ENFORCE(!operand.isComplex() && !min.isComplex() && !max.isComplex());
  return hal::clamp(ctx, operand, min, max);
}

}  // namespace spu::kernel::hlo
