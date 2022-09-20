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

#include "spu/device/pphlo/kernels/utils.h"

#include "spu/hal/test_util.h"

namespace spu::device {

bool getConditionValue(HalContext *ctx, const hal::Value &value) {
  YASL_ENFORCE(value.numel() == 1, "Condition value must be a scalar tensor.");
  YASL_ENFORCE(value.dtype() == DT_I1, "Expect bool, got {}", value.dtype());

  const auto public_val = hal::test::dump_public_as<bool>(ctx, value);
  return public_val.front();
}

xt::xarray<int64_t> getIndicies(HalContext *ctx, const hal::Value &value) {
  YASL_ENFORCE(value.isInt(), "indicies value must be integers.");
  YASL_ENFORCE(value.isPublic(), "indicies value must be public.");
  return hal::test::dump_public_as<int64_t>(ctx, value);
}

} // namespace spu::device
