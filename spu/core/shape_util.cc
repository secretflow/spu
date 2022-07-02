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

#include "spu/core/shape_util.h"

#include <numeric>
#include <vector>

#include "yasl/base/exception.h"

#include "spu/core/type_util.h"

namespace spu {

int64_t calcNumel(absl::Span<const int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
                         std::multiplies<>());
}

// Citation:
// https://github.com/xtensor-stack/xtensor-blas/blob/master/include/xtensor-blas/xlinalg.hpp
std::vector<int64_t> deduceDotShape(absl::Span<const int64_t> lhs,
                                    absl::Span<const int64_t> rhs) {
  // One side is scalar.
  if (lhs.empty() || rhs.empty()) {
    return lhs.empty() ? std::vector<int64_t>(rhs.begin(), rhs.end())
                       : std::vector<int64_t>(lhs.begin(), lhs.end());
  }

  // Vector dot product.
  if (lhs.size() == 1 && rhs.size() == 1) {
    YASL_ENFORCE_EQ(lhs[0], rhs[0],
                    "deduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);
    return {1};
  }

  if (lhs.size() == 2 && rhs.size() == 1) {
    // Matrix-times-vector product.
    YASL_ENFORCE_EQ(lhs[1], rhs[0],
                    "deduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);

    return {lhs[0]};
  } else if (lhs.size() == 1 && rhs.size() == 2) {
    // Matrix-times-vector product.
    YASL_ENFORCE_EQ(lhs[0], rhs[0],
                    "deduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);

    return {rhs[1]};
  } else if (lhs.size() == 2 && rhs.size() == 2) {
    // Matrix-product.
    YASL_ENFORCE_EQ(lhs[1], rhs[0],
                    "deduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);
    return {lhs[0], rhs[1]};
  } else {
    // If lhs is an N-D array and rhs is an M-D array (where M>=2), it is a sum
    // product over the last axis of lhs and the second-to-last axis of rhs:
    //    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    auto lhs_back = lhs.back();
    size_t rhs_match_dim = 0;

    // rhs may be vector.
    if (rhs.size() > 1) {
      rhs_match_dim = rhs.size() - 2;
    }

    YASL_ENFORCE_EQ(lhs_back, rhs[rhs_match_dim],
                    "deduceDotShape: shape mismatch: lhs={} ,rhs={}", lhs, rhs);

    int lhs_dim = static_cast<int>(lhs.size());
    int rhs_dim = static_cast<int>(rhs.size());

    int nd = lhs_dim + rhs_dim - 2;

    size_t j = 0;
    std::vector<int64_t> result(nd);

    for (int i = 0; i < lhs_dim - 1; ++i) {
      result[j++] = lhs[i];
    }

    for (int i = 0; i < rhs_dim - 2; ++i) {
      result[j++] = rhs[i];
    }

    if (rhs_dim > 1) {
      result[j++] = rhs.back();
    }

    return result;
  }
}

std::vector<int64_t> makeCompactStrides(absl::Span<const int64_t> shape) {
  std::vector<int64_t> strides(shape.size());
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

// This function assumes row major
int64_t flattenIndex(absl::Span<const int64_t> indices,
                     absl::Span<const int64_t> shape) {
  YASL_ENFORCE(indices.size() == shape.size());

  int64_t linear_idx = 0;
  int64_t stride = 1;
  for (int64_t idx = indices.size() - 1; idx >= 0; --idx) {
    linear_idx += indices[idx] * stride;
    stride *= shape[idx];
  }
  return linear_idx;
}

std::vector<int64_t> unflattenIndex(int64_t index,
                                    absl::Span<const int64_t> shape) {
  std::vector<int64_t> indices(shape.size(), 0);

  for (int64_t idx = indices.size() - 1; idx >= 0; --idx) {
    indices[idx] = index % shape[idx];
    index /= shape[idx];
  }
  return indices;
}

}  // namespace spu
