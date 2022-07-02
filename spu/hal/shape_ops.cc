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

#include "spu/hal/shape_ops.h"

#include "xtensor/xeval.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xstrides.hpp"
#include "yasl/base/exception.h"

#include "spu/core/ndarray_ref.h"
#include "spu/core/vectorize.h"
#include "spu/core/xt_helper.h"

namespace spu::hal {

namespace {

std::vector<int64_t> deducePadShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& edge_padding_low,
    const std::vector<int64_t>& edge_padding_high,
    const std::vector<int64_t>& interior_padding) {
  std::vector<int64_t> dims;
  YASL_ENFORCE(edge_padding_low.size() == input_shape.size());
  YASL_ENFORCE(edge_padding_high.size() == input_shape.size());
  YASL_ENFORCE(interior_padding.size() == input_shape.size());
  for (size_t i = 0; i < input_shape.size(); i++) {
    dims.emplace_back(edge_padding_low[i] + edge_padding_high[i] +
                      interior_padding[i] * (input_shape[i] - 1) +
                      input_shape[i]);
  }

  return dims;
}

// Adapted from:
// https://github.com/xtensor-stack/xtensor/blob/78aaac39143caa78da7c5c0734ccef957535f0c0/include/xtensor/xoperation.hpp#L877-L900
template <xt::layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class T>
inline auto AllIndices(const T& arr) {
  const auto& shape = arr.shape();
  using index_type = xt::xindex_type_t<typename T::shape_type>;
  using size_type = typename T::size_type;

  auto idx = xtl::make_sequence<index_type>(arr.dimension(), 0);
  std::vector<index_type> indices;

  size_type total_size = xt::compute_size(shape);
  for (size_type i = 0; i < total_size;
       i++, xt::detail::next_idx<L>(shape, idx)) {
    indices.push_back(idx);
  }
  return indices;
}

}  // namespace

Value transpose(HalContext* ctx, const Value& in,
                std::vector<int64_t> permutation) {
  SPU_TRACE_HAL(ctx, in);

  if (permutation.empty()) {
    permutation.resize(in.data().ndim());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::reverse(permutation.begin(), permutation.end());
  }

  // TODO(jint) dont touch membuf, manipulate strides for transpose.
  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    const auto& out =
        xt::eval(xt::transpose(xt_adapt<element_t>(in.data()), permutation));

    // TODO(jint) double-check xt strides convention.
    auto buf = std::make_shared<yasl::Buffer>(out.data(), out.size() * _kSize);
    return Value(
        {std::move(buf), in.storage_type(), out.shape(), out.strides(), 0},
        in.dtype());
  });
}

Value concatenate(HalContext* ctx, absl::Span<const Value> values,
                  const size_t& axis) {
  SPU_TRACE_HAL(ctx, axis);

  // Enforce all types are the same
  YASL_ENFORCE(
      std::all_of(values.begin() + 1, values.end(), [&](const Value& v) {
        return v.storage_type() == values.begin()->storage_type();
      }));

  // Enforce axis
  YASL_ENFORCE(std::all_of(values.begin(), values.end(), [&](const Value& v) {
    return static_cast<size_t>(axis) < v.shape().size() ||
           (v.shape().empty() && axis == 0);
  }));

  // Sanity shape
  for (size_t d = 0; d < values.front().shape().size(); ++d) {
    if (d == axis) {
      continue;
    }
    YASL_ENFORCE(
        std::all_of(values.begin() + 1, values.end(), [&](const Value& v) {
          return v.shape()[d] == values.front().shape()[d];
        }));
  }

  std::vector<int64_t> result_shape = values.front().shape();
  for (auto iter = values.begin() + 1; iter != values.end(); ++iter) {
    result_shape[axis] += iter->shape()[axis];
  }

  // Preallocate output buffer
  Value result({values.front().storage_type(), result_shape},
               values.front().dtype());

  int64_t b_dimension_offset = 0;
  for (const auto& v : values) {
    std::vector<int64_t> from_indicies(result_shape.size(), 0);
    std::vector<int64_t> to_indicies(result_shape.size(), 0);
    do {
      to_indicies = from_indicies;
      to_indicies[axis] += b_dimension_offset;
      result.copyElementFrom(v, from_indicies, to_indicies);
    } while (bumpIndices<int64_t>(v.shape(), absl::MakeSpan(from_indicies)));
    b_dimension_offset += v.shape()[axis];
  }

  return result;
}

Value slice(HalContext* ctx, const Value& in,
            absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> end_indices,
            absl::Span<const int64_t> strides) {
  SPU_TRACE_HAL(ctx, in, start_indices, end_indices, strides);

  YASL_ENFORCE(in.shape().size() == start_indices.size());
  YASL_ENFORCE(in.shape().size() == end_indices.size());
  YASL_ENFORCE(strides.empty() || (in.shape().size() == strides.size()));

  xt::xstrided_slice_vector sv;
  for (size_t idx = 0; idx < in.shape().size(); ++idx) {
    sv.push_back(xt::range(start_indices[idx], end_indices[idx],
                           strides.empty() ? 1 : strides[idx]));
  }

  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    const auto& out = xt::strided_view(xt_adapt<element_t>(in.data()), sv);

    return Value(NdArrayRef(in.data().buf(), in.storage_type(), out.shape(),
                            out.strides(), out.data_offset() * in.elsize()),
                 in.dtype());
  });
}

Value reshape(HalContext* ctx, const Value& in,
              const std::vector<int64_t>& to_shape) {
  SPU_TRACE_HAL(ctx, in, to_shape);

  YASL_ENFORCE(calcNumel(in.shape()) == calcNumel(to_shape),
               "reshape, numel mismatch, lhs={}, rhs={}", in.shape(), to_shape);

  // TODO(jint) dont touch membuf, manipulate strides for transpose.
  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    const auto& out =
        xt::eval(xt::reshape_view(xt_adapt<element_t>(in.data()), to_shape));

    auto buf = std::make_shared<yasl::Buffer>(out.data(), out.size() * _kSize);
    return Value(
        {std::move(buf), in.storage_type(), out.shape(), out.strides(), 0},
        in.dtype());
  });
}

Value broadcast_to(HalContext* ctx, const Value& in,
                   const std::vector<int64_t>& to_shape,
                   const std::vector<size_t>& in_dims) {
  SPU_TRACE_HAL(ctx, in, to_shape);

  if (in.shape() == to_shape) {
    return in;
  }

  Value operand;
  if (!in_dims.empty() && (in.shape().size() != to_shape.size())) {
    // Needs a reshape
    std::vector<int64_t> reshape_to(to_shape.size(), 1);
    for (size_t idx = 0; idx < in_dims.size(); ++idx) {
      reshape_to[in_dims[idx]] = in.shape()[idx];
    }
    operand = hal::reshape(ctx, in, reshape_to);
  } else {
    operand = in;
  }

  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    NdArrayRef ret(in.data().eltype(), to_shape);
    xt_mutable_adapt<element_t>(ret) =
        xt::eval(xt::broadcast(xt_adapt<element_t>(operand.data()), to_shape));
    return Value(ret, in.dtype());
  });
}

Value reverse(HalContext* ctx, const Value& in,
              const std::vector<size_t>& dimensions) {
  SPU_TRACE_HAL(ctx, in, dimensions);
  return DISPATCH_ALL_ELSIZE(in.elsize(), [&]() -> Value {
    xt::xarray<element_t> expr = xt_adapt<element_t>(in.data());

    for (const auto dim : dimensions) {
      expr = xt::flip(expr, dim);
    }

    NdArrayRef ret(in.data().eltype(), in.shape());
    xt_mutable_adapt<element_t>(ret) = xt::eval(expr);
    return Value(ret, in.dtype());
  });
}

Value pad(HalContext* ctx, const Value& in, const Value& padding_value,
          const std::vector<int64_t>& edge_padding_low,
          const std::vector<int64_t>& edge_padding_high,
          const std::vector<int64_t>& interior_padding) {
  YASL_ENFORCE(in.storage_type() == padding_value.storage_type());
  Value result =
      broadcast_to(ctx, padding_value,
                   deducePadShape(in.shape(), edge_padding_low,
                                  edge_padding_high, interior_padding));

  const auto& result_shape = result.shape();
  const auto& input_shape = in.shape();

  const int64_t rank = input_shape.size();

  std::vector<int64_t> result_base(result_shape.size(), 0);
  std::vector<int64_t> input_base(input_shape.size(), 0);

  std::vector<int64_t> result_incr(result_shape.size(), 1);
  std::vector<int64_t> input_incr(input_shape.size(), 1);

  std::vector<int64_t> result_count = result_shape;
  std::vector<int64_t> input_count = input_shape;

  for (int64_t idx = 0; idx < rank; ++idx) {
    const auto padding_low = edge_padding_low[idx];
    const auto padding_high = edge_padding_high[idx];
    const auto in_padding = interior_padding[idx];
    if (padding_low < 0) {
      input_base[idx] =
          std::ceil(-padding_low / static_cast<float>(1 + in_padding));
      result_base[idx] = (-padding_low % (1 + in_padding));
    } else {
      result_base[idx] += padding_low;
    }
    if (padding_high < 0) {
      input_count[idx] -=
          std::ceil(-padding_high / static_cast<float>(1 + in_padding));
      result_count[idx] -= (-padding_high % (1 + in_padding));
    } else {
      result_count[idx] += padding_high;
    }
    // interior padding cannot be negative
    result_incr[idx] += in_padding;
  }

  int64_t n = -1;
  std::vector<int64_t> input_index(input_base.begin(), input_base.end());
  std::vector<int64_t> result_index(result_base.begin(), result_base.end());

  auto copy_element = [&](absl::Span<const int64_t> result_index,
                          absl::Span<const int64_t> input_index) {
    result.copyElementFrom(in, input_index, result_index);
  };

  while (n < rank) {
    copy_element(result_index, input_index);
    // Increments dimensions in minor to major order.
    for (n = 0; n < rank; ++n) {
      input_index[n] += input_incr[n];
      result_index[n] += result_incr[n];
      if ((input_index[n] < input_base[n] + input_count[n]) &&
          (result_index[n] < result_base[n] + result_count[n])) {
        break;
      }
      input_index[n] = input_base[n];
      result_index[n] = result_base[n];
    }
  }

  return result;
}

}  // namespace spu::hal
