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

#include "libspu/mpc/standard_shape/kernels.h"

#include <set>

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc::standard_shape {

// Compact threshold heuristic, try to make it same as L1 cache size
#define COMPACT_THRESHOLD (32 * 1024)  // 32K

SPU_ALWAYS_INLINE NdArrayRef _try_compact(const NdArrayRef& in) {
  // If in data is not compact after some shape ops and small enough, make it
  // compact
  if (in.numel() * in.elsize() <= COMPACT_THRESHOLD && !in.isCompact()) {
    return in.clone();
  }
  return in;
}

NdArrayRef Broadcast::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const Shape& to_shape, const Axes& in_dims) const {
  return in.broadcast_to(to_shape, in_dims);
}

NdArrayRef Reshape::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         const Shape& to_shape) const {
  return _try_compact(in.reshape(to_shape));
}

NdArrayRef ExtractSlice::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                              const Index& start, const Index& end,
                              const Strides& strides) const {
  return _try_compact(in.slice(start, end, strides));
}

NdArrayRef UpdateSlice::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                             const NdArrayRef& update,
                             const Index& start) const {
  SPU_ENFORCE(in.eltype() == update.eltype(),
              "Element type mismatch, in = {}, update ={}", in.eltype(),
              update.eltype());

  auto ret = in.clone();
  ret.update_slice(update, start);
  return ret;
}

NdArrayRef Transpose::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const Axes& permutation) const {
  Axes perm = permutation;
  if (perm.empty()) {
    // by default, transpose the data in reverse order.
    perm.resize(in.shape().size());
    std::iota(perm.rbegin(), perm.rend(), 0);
  }

  // sanity check.
  SPU_ENFORCE_EQ(perm.size(), in.shape().size());
  std::set<int64_t> uniq(perm.begin(), perm.end());
  SPU_ENFORCE_EQ(uniq.size(), perm.size(), "perm={} is not unique", perm);

  // fast path, if identity permutation, return it.
  Axes no_perm(in.shape().size());
  std::iota(no_perm.begin(), no_perm.end(), 0);
  if (perm == no_perm) {
    return in;
  }

  return _try_compact(in.transpose(perm));
}

NdArrayRef Reverse::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         const Axes& dimensions) const {
  return in.reverse(dimensions);
}

NdArrayRef Fill::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                      const Shape& to_shape) const {
  return in.expand(to_shape);
}

NdArrayRef Pad::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     const NdArrayRef& padding_value,
                     const Sizes& edge_padding_low,
                     const Sizes& edge_padding_high,
                     const Sizes& interior_padding) const {
  SPU_ENFORCE(in.eltype() == padding_value.eltype(),
              "Element type mismatch, in = {}, pad_value ={}", in.eltype(),
              padding_value.eltype());
  return in.pad(padding_value, edge_padding_low, edge_padding_high,
                interior_padding);
}

NdArrayRef Concate::proc(KernelEvalContext* ctx,
                         const std::vector<NdArrayRef>& values,
                         int64_t axis) const {
  return values.front().concatenate(
      absl::MakeSpan(&values[1], values.size() - 1), axis);
}

}  // namespace spu::mpc::standard_shape
