// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/geometrical.h"

#include "libspu/kernel/hal/complex.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

spu::Value Transpose(SPUContext *ctx, const spu::Value &in,
                     const Axes &permutation) {
  if (in.isComplex()) {
    auto r = hal::transpose(ctx, hal::real(ctx, in), permutation);
    auto i = hal::transpose(ctx, hal::imag(ctx, in), permutation);
    return hal::complex(ctx, r, i);
  }
  return hal::transpose(ctx, in, permutation);
}

spu::Value Broadcast(SPUContext *ctx, const spu::Value &in,
                     const Shape &to_shape, const Axes &in_dims) {
  if (in.isComplex()) {
    auto r = hal::broadcast_to(ctx, hal::real(ctx, in), to_shape, in_dims);
    auto i = hal::broadcast_to(ctx, hal::imag(ctx, in), to_shape, in_dims);
    return hal::complex(ctx, r, i);
  }
  return hal::broadcast_to(ctx, in, to_shape, in_dims);
}

spu::Value Reshape(SPUContext *ctx, const spu::Value &in,
                   const Shape &to_shape) {
  if (in.isComplex()) {
    auto r = hal::reshape(ctx, hal::real(ctx, in), to_shape);
    auto i = hal::reshape(ctx, hal::imag(ctx, in), to_shape);
    return hal::complex(ctx, r, i);
  }
  return hal::reshape(ctx, in, to_shape);
}

spu::Value Concatenate(SPUContext *ctx, absl::Span<const spu::Value> operands,
                       int64_t axis) {
  if (operands.front().isComplex()) {
    std::vector<spu::Value> r_operands(operands.size());
    std::vector<spu::Value> i_operands(operands.size());
    for (size_t idx = 0; idx < operands.size(); ++idx) {
      r_operands[idx] = hal::real(ctx, operands[idx]);
      i_operands[idx] = hal::imag(ctx, operands[idx]);
    }

    auto r = hal::concatenate(ctx, r_operands, axis);
    auto i = hal::concatenate(ctx, i_operands, axis);
    return hal::complex(ctx, r, i);
  }
  return hal::concatenate(ctx, operands, axis);
}

spu::Value Slice(SPUContext *ctx, const spu::Value &in, const Index &start,
                 const Index &end, const Strides &strides) {
  if (in.isComplex()) {
    auto r = hal::slice(ctx, hal::real(ctx, in), start, end, strides);
    auto i = hal::slice(ctx, hal::imag(ctx, in), start, end, strides);
    return hal::complex(ctx, r, i);
  }
  return hal::slice(ctx, in, start, end, strides);
}

spu::Value Pad(SPUContext *ctx, const spu::Value &in,
               const spu::Value &pad_value, const Sizes &edge_low,
               const Sizes &edge_high, const Sizes &inner) {
  if (in.isComplex()) {
    auto r = hal::pad(ctx, hal::real(ctx, in), hal::real(ctx, pad_value),
                      edge_low, edge_high, inner);
    auto i = hal::pad(ctx, hal::imag(ctx, in), hal::imag(ctx, pad_value),
                      edge_low, edge_high, inner);
    return hal::complex(ctx, r, i);
  }
  return hal::pad(ctx, in, pad_value, edge_low, edge_high, inner);
}

spu::Value Reverse(SPUContext *ctx, const spu::Value &in, const Axes &dims) {
  if (in.isComplex()) {
    auto r = hal::reverse(ctx, hal::real(ctx, in), dims);
    auto i = hal::reverse(ctx, hal::imag(ctx, in), dims);
    return hal::complex(ctx, r, i);
  }
  return hal::reverse(ctx, in, dims);
}

}  // namespace spu::kernel::hlo
