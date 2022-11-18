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

#include "spu/kernel/hal/shape_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "spu/core/ndarray_ref.h"
#include "spu/core/parallel_utils.h"
#include "spu/core/shape_util.h"
#include "spu/core/vectorize.h"

namespace spu::kernel::hal {

namespace {

std::vector<int64_t> deducePadShape(
    absl::Span<const int64_t> input_shape,
    absl::Span<const int64_t> edge_padding_low,
    absl::Span<const int64_t> edge_padding_high,
    absl::Span<const int64_t> interior_padding) {
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

}  // namespace

Value transpose(HalContext* ctx, const Value& in,
                absl::Span<const int64_t> permutation) {
  SPU_TRACE_HAL_DISP(ctx, in);

  std::vector<int64_t> perm(in.shape().size());
  if (permutation.empty()) {
    for (size_t i = 0; i < perm.size(); ++i) {
      perm[i] = static_cast<int64_t>(in.shape().size()) - 1 - i;
    }
  } else {
    std::vector<int64_t> reverse_permutation(in.shape().size(), -1);
    YASL_ENFORCE(permutation.size() == in.shape().size(),
                 "axes don't match array, permutation = {}, input shape = {}",
                 fmt::join(permutation, "x"), fmt::join(in.shape(), "x"));

    for (size_t i = 0; i < permutation.size(); i++) {
      auto axis = permutation[i];
      YASL_ENFORCE(reverse_permutation[axis] == -1,
                   "repeated axis in transpose");
      reverse_permutation[axis] = i;
      perm[i] = axis;
    }
  }

  std::vector<int64_t> ret_shape(in.shape().size());
  std::vector<int64_t> ret_strides(in.strides().size());

  for (size_t i = 0; i < in.shape().size(); i++) {
    ret_shape[i] = in.shape()[perm[i]];
    ret_strides[i] = in.strides()[perm[i]];
  }

  // compact clone is a rather expensive memory operation.
  // To prevent transposed value being cloned multiple times in later ops, clone
  // the value here.
  auto transposed = NdArrayRef{in.data().buf(), in.storage_type(), ret_shape,
                               ret_strides, in.data().offset()}
                        .clone();
  return Value(transposed, in.dtype());
}

Value slice(HalContext* ctx, const Value& in,
            absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> end_indices,
            absl::Span<const int64_t> strides) {
  SPU_TRACE_HAL_DISP(ctx, in, start_indices, end_indices, strides);

  YASL_ENFORCE(in.shape().size() == start_indices.size());
  YASL_ENFORCE(in.shape().size() == end_indices.size());
  YASL_ENFORCE(strides.empty() || (in.shape().size() == strides.size()));

  std::vector<int64_t> new_shape(in.shape().size(), 0);
  std::vector<int64_t> new_strides(in.strides());
  for (size_t idx = 0; idx < in.shape().size(); ++idx) {
    YASL_ENFORCE(end_indices[idx] <= in.shape()[idx],
                 "Slice end at axis {} = {} is larger than input shape {}", idx,
                 end_indices[idx], in.shape()[idx]);
    new_shape[idx] = end_indices[idx] - start_indices[idx];
    if (!strides.empty()) {
      auto n = new_shape[idx] / strides[idx];
      auto q = new_shape[idx] % strides[idx];
      new_shape[idx] = n + static_cast<int64_t>(q != 0);
      new_strides[idx] *= strides[idx];
    }
  }

  return Value(
      NdArrayRef(
          in.data().buf(), in.storage_type(), new_shape, new_strides,
          &in.data().at(start_indices) - in.data().buf()->data<std::byte>()),
      in.dtype());
}

// Reference:
// https://github.com/numpy/numpy/blob/c652fcbd9c7d651780ea56f078c8609932822cf7/numpy/core/src/multiarray/shape.c#L371
bool attempt_nocopy_reshape(const Value& old,
                            absl::Span<const int64_t> new_shape,
                            std::vector<int64_t>& new_strides) {
  size_t oldnd;
  std::vector<int64_t> olddims(old.shape().size());
  std::vector<int64_t> oldstrides(old.strides().size());
  size_t oi;
  size_t oj;
  size_t ok;
  size_t ni;
  size_t nj;
  size_t nk;

  oldnd = 0;
  /*
   * Remove axes with dimension 1 from the old array. They have no effect
   * but would need special cases since their strides do not matter.
   */
  for (oi = 0; oi < old.shape().size(); oi++) {
    if (old.shape()[oi] != 1) {
      olddims[oldnd] = old.shape()[oi];
      oldstrides[oldnd] = old.strides()[oi];
      oldnd++;
    }
  }

  /* oi to oj and ni to nj give the axis ranges currently worked with */
  oi = 0;
  oj = 1;
  ni = 0;
  nj = 1;
  while (ni < new_shape.size() && oi < oldnd) {
    auto np = new_shape[ni];
    auto op = olddims[oi];

    while (np != op) {
      if (np < op) {
        /* Misses trailing 1s, these are handled later */
        np *= new_shape[nj++];
      } else {
        op *= olddims[oj++];
      }
    }

    /* Check whether the original axes can be combined */
    for (ok = oi; ok < oj - 1; ok++) {
      if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
        /* not contiguous enough */
        return false;
      }
    }

    /* Calculate new strides for all axes currently worked with */
    new_strides[nj - 1] = oldstrides[oj - 1];
    for (nk = nj - 1; nk > ni; nk--) {
      new_strides[nk - 1] = new_strides[nk] * new_shape[nk];
    }

    ni = nj++;
    oi = oj++;
  }

  for (size_t idx = 0; idx < new_shape.size(); ++idx) {
    if (new_shape[idx] == 1) {
      // During attempt_nocopy_reshape strides for 1 sized dimensions are not
      // set to 0, which can be a problem if this value is later broadcasted
      // in this dimension, so force set to 0 here
      new_strides[idx] = 0;
    }
  }

  return true;
}

Value reshape(HalContext* ctx, const Value& in,
              absl::Span<const int64_t> to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  // Nothing to reshape
  if (in.shape() == to_shape) {
    return in;
  }

  YASL_ENFORCE(calcNumel(in.shape()) == calcNumel(to_shape),
               "reshape, numel mismatch, lhs={}, rhs={}", in.shape(), to_shape);

  std::vector<int64_t> new_strides(to_shape.size(), 0);
  if (attempt_nocopy_reshape(in, to_shape, new_strides)) {
    return Value({in.data().buf(), in.storage_type(), to_shape, new_strides,
                  in.data().offset()},
                 in.dtype());
  }

  auto compact_clone = in.data().clone();
  return Value({compact_clone.buf(), in.storage_type(), to_shape}, in.dtype());
}

Value broadcast_to(HalContext* ctx, const Value& in,
                   absl::Span<const int64_t> to_shape,
                   absl::Span<const int64_t> in_dims) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  std::vector<int64_t> new_strides(to_shape.size(), 0);

  if (!in_dims.empty()) {
    for (size_t idx = 0; idx < in_dims.size(); ++idx) {
      new_strides[in_dims[idx]] = in.strides()[idx];
    }
  } else {
    for (size_t idx = 0; idx < in.strides().size(); ++idx) {
      new_strides.at(new_strides.size() - 1 - idx) =
          in.strides().at(in.strides().size() - 1 - idx);
    }
  }

  return Value(NdArrayRef(in.data().buf(), in.data().eltype(), to_shape,
                          new_strides, in.data().offset()),
               in.dtype());
}

Value reverse(HalContext* ctx, const Value& in,
              absl::Span<const int64_t> dimensions) {
  SPU_TRACE_HAL_DISP(ctx, in, dimensions);

  std::vector<int64_t> new_strides = in.strides();
  int64_t el_offset = 0;

  for (int64_t axis : dimensions) {
    YASL_ENFORCE(axis < static_cast<int64_t>(in.shape().size()));
    new_strides[axis] *= -1;
    el_offset += in.strides()[axis] * (in.shape()[axis] - 1);
  }

  return Value(
      NdArrayRef(in.data().buf(), in.data().eltype(), in.shape(), new_strides,
                 in.data().offset() + el_offset * in.elsize()),
      in.dtype());
}

Value pad(HalContext* ctx, const Value& in, const Value& padding_value,
          absl::Span<const int64_t> edge_padding_low,
          absl::Span<const int64_t> edge_padding_high,
          absl::Span<const int64_t> interior_padding) {
  YASL_ENFORCE(in.storage_type() == padding_value.storage_type());
  Value result = expand(ctx, padding_value,
                        deducePadShape(in.shape(), edge_padding_low,
                                       edge_padding_high, interior_padding));

  const auto& result_shape = result.shape();
  const auto& input_shape = in.shape();

  auto elsize = result.elsize();

  yasl::parallel_for(0, in.numel(), 1024, [&](int64_t begin, int64_t end) {
    std::vector<int64_t> unflatten = unflattenIndex(begin, input_shape);

    std::vector<int64_t> target_index(result_shape.size());
    for (int64_t idx = begin; idx < end; ++idx) {
      bool valid = true;
      for (size_t i = 0; i < unflatten.size(); ++i) {
        // Interior padding occurs logically before edge padding, so in the case
        // of negative edge padding elements are removed from the
        // interior-padded operand.
        target_index[i] =
            edge_padding_low[i] + unflatten[i] * (interior_padding[i] + 1);

        // Account for negative low and high padding: skip assignment if the
        // any target index is out of range.
        if (!(target_index[i] >= 0 && target_index[i] < result_shape[i])) {
          valid = false;
          break;
        }
      }
      if (valid) {
        result.copyElementFrom(in, unflatten, target_index, elsize);
      }
      bumpIndices<int64_t>(in.shape(), absl::MakeSpan(unflatten));
    }
  });

  return result;
}

Value expand(HalContext* ctx, const Value& in,
             absl::Span<const int64_t> to_shape) {
  YASL_ENFORCE(in.numel() == 1, "Only support expanding scalar");
  Value ret({in.data().eltype(), to_shape}, in.dtype());
  // compute number of elements need to copy
  size_t numel = ret.numel();
  size_t num_bytes = numel * in.elsize();
  size_t bytes_copied = in.elsize();

  // Copy first element
  std::memcpy(ret.data().data(), in.data().data(), in.elsize());

  while (bytes_copied != num_bytes) {
    size_t copy_size = std::min(bytes_copied, num_bytes - bytes_copied);
    std::memcpy(static_cast<char*>(ret.data().data()) + bytes_copied,
                ret.data().data(), copy_size);
    bytes_copied += copy_size;
  }
  return ret;
}

}  // namespace spu::kernel::hal
