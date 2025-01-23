// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/core/shape.h"

namespace spu {

bool Index::inBounds(const Shape& bounds) const {
  if (size() != bounds.size()) {
    return false;
  }

  for (size_t idx = 0; idx < bounds.size(); ++idx) {
    if ((*this)[idx] < 0 || (*this)[idx] >= bounds[idx]) {
      return false;
    }
  }
  return true;
}

Strides makeCompactStrides(const Shape& shape) {
  Strides strides(shape.size());
  const size_t size = shape.size();
  for (size_t dim = size; dim > 0; dim--) {
    strides[dim - 1] = dim == size ? 1 : strides[dim] * shape[dim];
  }
  // This follows the xtensor style, @jint I think both 0 or `default value`
  // should be OK.
  for (size_t dim = 0; dim < size; dim++) {
    if (shape[dim] == 1) {
      strides[dim] = 0;
    }
  }
  return strides;
}

int64_t flattenIndex(const Index& index, const Shape& shape) {
  SPU_ENFORCE(index.size() == shape.size());

  int64_t linear_idx = 0;
  int64_t stride = 1;
  for (int64_t idx = index.size() - 1; idx >= 0; --idx) {
    linear_idx += index[idx] * stride;
    stride *= shape[idx];
  }
  return linear_idx;
}

Index unflattenIndex(int64_t index, const Shape& shape) {
  Index unflattened(shape.size());
  for (int64_t idx = unflattened.size() - 1; idx >= 0; --idx) {
    unflattened[idx] = index % shape[idx];
    index /= shape[idx];
  }
  return unflattened;
}

}  // namespace spu
