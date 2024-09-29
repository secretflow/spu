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

#include <cstdint>

#include "libspu/core/memref.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

/// the broadcast function
// @param in, the input
// @param to_shape, the target shape
MemRef broadcast_to(SPUContext* ctx, const MemRef& in, const Shape& to_shape,
                    const Axes& in_dims = {});

/// the reshape function
// @param in, the input
// @param to_shape, the target shape
MemRef reshape(SPUContext* ctx, const MemRef& in, const Shape& to_shape);

/// the slice function
// @param input, the param
// @param offsets, the start indices
// @param sizes, sizes of slice
// @param strides, the strides
MemRef slice(SPUContext* ctx, const MemRef& input, const Index& offsets,
             const Shape& sizes, const Strides& strides = {});

/// This is a special slice for single element at indices
// @returns a array with empty shape (scalar)
MemRef slice_scalar_at(SPUContext* ctx, const MemRef& input,
                       const Index& indices);

// update a block of in with update, start_indices is postion at in
MemRef insert_slice(SPUContext* ctx, const MemRef& in, const MemRef& update,
                    const Index& offsets, const Strides& strides,
                    bool prefer_in_place);

/// the transpose function
// @param in, the param
MemRef transpose(SPUContext* ctx, const MemRef& in,
                 const Axes& permutation = {});

//// the reverse function
// @param in, the param
// @param dimensions, dimensions to reverse
MemRef reverse(SPUContext* ctx, const MemRef& in, const Axes& dimensions);

/// Expand a scalar into to_shape.
/// Compare with broadcast, expand actually reallocates and assign memory
MemRef expand(SPUContext* ctx, const MemRef& in, const Shape& to_shape);

//// the pad function
// @param in, the param
// @param padding_value, to fill in the added padding
// @param edge_padding_low, the amount of padding added at the
//        low-end (next to index 0) of each dimension
// @param edge_padding_high, the amount of padding added at the high-end
//        (next to the highest index) of each dimension
MemRef pad(SPUContext* ctx, const MemRef& in, const MemRef& padding_value,
           const Sizes& edge_padding_low, const Sizes& edge_padding_high,
           const Sizes& interior_padding = {});

/// the concatenate function
// @param first, the first param
// @param second, the second param
// @param axis, the axis
MemRef concatenate(SPUContext* ctx, const std::vector<MemRef>& values,
                   int64_t axis);

//////////////////////////////////////////////////////////////////////////////
// Shape utils
//////////////////////////////////////////////////////////////////////////////

/// the squeeze function, i.e., removes dimensions of size 1 from the shape of a
/// tensor.
// @param in, the input
// @param dim, the dimension to be squeezed
MemRef squeeze(SPUContext* ctx, const MemRef& in, int64_t dim = 0);

/// the unsqueeze function, i.e., expands a tensor with a length 1 axis inserted
/// at index axis.
// @param in, the input
// @param dim, the dimension to be unsqueezed
MemRef unsqueeze(SPUContext* ctx, const MemRef& in, int64_t dim = 0);

}  // namespace spu::kernel::hal
