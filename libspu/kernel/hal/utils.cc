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

#include "libspu/kernel/hal/utils.h"

namespace spu::kernel::hal {

Value squeeze(SPUContext* ctx, const Value& in, int64_t dim) {
  SPU_ENFORCE(dim >= 0 && dim < in.shape().ndim(),
              "input shape {} and squeezing dim {} are mismatched", in.shape(),
              dim);
  Shape new_shape = in.shape();
  if (new_shape[dim] == 1) {
    new_shape.erase(new_shape.begin() + dim);
    return hal::reshape(ctx, in, new_shape);
  }
  return in;
}

Value unsqueeze(SPUContext* ctx, const Value& in, int64_t dim) {
  SPU_ENFORCE(dim >= 0 && dim <= in.shape().ndim(),
              "input shape {} and unsqueezing dim {} are mismatched",
              in.shape(), dim);
  Shape new_shape = in.shape();
  new_shape.insert(new_shape.begin() + dim, 1);
  return hal::reshape(ctx, in, new_shape);
}

}  // namespace spu::kernel::hal
