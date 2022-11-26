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

#pragma once

#include <vector>

#include "xtensor/xeval.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xsort.hpp"

#include "spu/core/ndarray_ref.h"
#include "spu/core/xt_helper.h"
#include "spu/kernel/context.h"
#include "spu/kernel/hal/shape_ops.h"
#include "spu/kernel/value.h"

namespace spu::kernel::hal {

/// FIXME: Remove these hacky templates once we get rid of xtensor here
template <size_t S>
struct element_t_s {
  std::array<std::byte, S> buf;
  // xtensor uses operator+ to compute type promotion rule of container element
  // So we provides a empty + to make it happy
  element_t_s operator+(const element_t_s& /*unused*/) { return *this; }
};

#define __CASE_SIZE(SIZE, ...)                       \
  case (SIZE): {                                     \
    using element_t = element_t_s<SIZE>;             \
    [[maybe_unused]] constexpr size_t _kSize = SIZE; \
    return __VA_ARGS__();                            \
  }

#define DISPATCH_ALL_ELSIZE(SIZE, ...)                         \
  [&] {                                                        \
    switch (SIZE) {                                            \
      __CASE_SIZE(1, __VA_ARGS__)                              \
      __CASE_SIZE(2, __VA_ARGS__)                              \
      __CASE_SIZE(4, __VA_ARGS__)                              \
      __CASE_SIZE(8, __VA_ARGS__)                              \
      __CASE_SIZE(16, __VA_ARGS__)                             \
      __CASE_SIZE(32, __VA_ARGS__)                             \
      default:                                                 \
        YACL_THROW("un-implemented for elment_size={}", SIZE); \
    }                                                          \
  }()

// NOTE(junfeng): idea is quite similar to argsort in spu/kernel/hal/sort.h.
template <class E>
Value permute(HalContext* ctx, const Value& x, size_t axis,
              const xt::xexpression<E>& permutation) {
  const size_t dimension = x.shape().size();

  const auto& dpermutation = permutation.derived_cast();

  const auto& x_data = x.data();
  if (dimension == 1) {
    Value result({x_data.eltype(), x.shape()}, x.dtype());

    for (int64_t i = 0; i < x.numel(); i++) {
      result.copyElementFrom(x, {(int64_t)dpermutation(i)}, {i});
    }

    return result;
  }

  if (axis < dimension - 1) {
    xt::dynamic_shape<std::size_t> perm;
    xt::dynamic_shape<std::size_t> reverse_perm;
    std::tie(perm, reverse_perm) = xt::detail::get_permutations(
        dpermutation.dimension(), axis, dpermutation.layout());

    auto permutation_t = xt::eval(xt::transpose(dpermutation, perm));

    const auto& x_data = x.data();
    return DISPATCH_ALL_ELSIZE(x_data.elsize(), [&]() -> Value {
      auto x_t = xt::eval(xt::transpose(xt_adapt<element_t>(x_data), perm));
      std::vector<int64_t> ret_shape{x_t.shape().begin(), x_t.shape().end()};
      NdArrayRef ret(x_data.eltype(), ret_shape);
      xt_mutable_adapt<element_t>(ret) = xt::empty<element_t>(ret_shape);

      std::size_t n_iters =
          std::accumulate(ret_shape.begin(), ret_shape.end() - 1,
                          std::size_t(1), std::multiplies<>());
      std::ptrdiff_t data_secondary_stride = ret_shape.back();
      auto x_ptr = x_t.data();
      auto permutation_ptr = permutation_t.data();
      auto ret_ptr = static_cast<element_t*>(ret.data());

      for (std::size_t i = 0; i < n_iters; i++, x_ptr += data_secondary_stride,
                       permutation_ptr += data_secondary_stride,
                       ret_ptr += data_secondary_stride) {
        for (std::ptrdiff_t j = 0; j < data_secondary_stride; j++) {
          std::memcpy(
              ret_ptr + j,
              x_ptr + static_cast<std::ptrdiff_t>(*(permutation_ptr + j)),
              sizeof(element_t));
        }
      }

      return transpose(ctx, Value(ret, x.dtype()),
                       absl::MakeSpan((const int64_t*)reverse_perm.data(),
                                      reverse_perm.size()));
    });
  }

  return DISPATCH_ALL_ELSIZE(x.data().elsize(), [&]() -> Value {
    auto ret_shape = x.shape();
    NdArrayRef ret(x.data().eltype(), ret_shape);
    xt_mutable_adapt<element_t>(ret) = xt::empty<element_t>(ret_shape);

    std::size_t n_iters =
        std::accumulate(ret_shape.begin(), ret_shape.end() - 1, std::size_t(1),
                        std::multiplies<>());
    std::ptrdiff_t data_secondary_stride = ret_shape[axis];
    auto x_ptr = static_cast<const element_t*>(x.data().data());
    auto permutation_ptr = dpermutation.data();
    auto ret_ptr = static_cast<element_t*>(ret.data());

    for (std::size_t i = 0; i < n_iters; i++, x_ptr += data_secondary_stride,
                     permutation_ptr += data_secondary_stride,
                     ret_ptr += data_secondary_stride) {
      for (std::ptrdiff_t j = 0; j < data_secondary_stride; j++) {
        std::memcpy(ret_ptr + j,
                    x_ptr + static_cast<std::ptrdiff_t>(*(permutation_ptr + j)),
                    sizeof(element_t));
      }
    }

    return Value(ret, x.dtype());
  });
}

}  // namespace spu::kernel::hal
