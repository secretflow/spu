// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/utils.h"

#include "libspu/kernel/hal/public_helper.h"

namespace spu::kernel {

bool getBooleanValue(SPUContext *ctx, const spu::Value &value) {
  SPU_ENFORCE(value.numel() == 1, "Condition value must be a scalar tensor.");
  SPU_ENFORCE(value.dtype() == DT_I1, "Expect bool, got {}", value.dtype());
  SPU_ENFORCE(value.isPublic(), "Expect public value");

  const auto public_val = kernel::hal::dump_public_as<bool>(ctx, value);
  return public_val.front();
}

int32_t getI32Value(SPUContext *ctx, const spu::Value &value) {
  SPU_ENFORCE(value.numel() == 1, "Index value must be a scalar tensor.");
  SPU_ENFORCE(value.dtype() == DT_I32, "Expect bool, got {}", value.dtype());
  SPU_ENFORCE(value.isPublic(), "Expect public value");

  const auto public_val = kernel::hal::dump_public_as<int32_t>(ctx, value);
  return public_val.front();
}

xt::xarray<int64_t> getIndices(SPUContext *ctx, const spu::Value &value) {
  SPU_ENFORCE(value.isInt(), "indices value must be integers.");
  SPU_ENFORCE(value.isPublic(), "indices value must be public.");
  return kernel::hal::dump_public_as<int64_t>(ctx, value);
}

}  // namespace spu::kernel
