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

#include "libspu/kernel/hal/prot_wrapper.h"

#include <cstddef>
#include <vector>

#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/api.h"

namespace spu::kernel::hal {

#define MAP_UNARY_OP(NAME)                          \
  Value _##NAME(SPUContext* ctx, const Value& in) { \
    SPU_TRACE_HAL_DISP(ctx, in);                    \
    return mpc::NAME(ctx, in);                      \
  }

#define MAP_SHIFT_OP(NAME)                                             \
  Value _##NAME(SPUContext* ctx, const Value& in, const Sizes& bits) { \
    SPU_TRACE_HAL_DISP(ctx, in, bits);                                 \
    auto ret = mpc::NAME(ctx, in, bits);                               \
    return ret;                                                        \
  }

#define MAP_BITREV_OP(NAME)                                                   \
  Value _##NAME(SPUContext* ctx, const Value& in, size_t start, size_t end) { \
    SPU_TRACE_HAL_DISP(ctx, in, start, end);                                  \
    auto ret = mpc::NAME(ctx, in, start, end);                                \
    return ret;                                                               \
  }

#define MAP_BINARY_OP(NAME)                                           \
  Value _##NAME(SPUContext* ctx, const Value& x, const Value& y) {    \
    SPU_TRACE_HAL_DISP(ctx, x, y);                                    \
    SPU_ENFORCE(x.shape() == y.shape(), "shape mismatch: x={}, y={}", \
                x.shape(), y.shape());                                \
    auto ret = mpc::NAME(ctx, x, y);                                  \
    return ret;                                                       \
  }

#define MAP_MMUL_OP(NAME)                                          \
  Value _##NAME(SPUContext* ctx, const Value& x, const Value& y) { \
    SPU_TRACE_HAL_DISP(ctx, x, y);                                 \
    auto ret = mpc::NAME(ctx, x, y);                               \
    return ret;                                                    \
  }

Type _common_type_s(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_HAL_DISP(ctx, a, b);
  return mpc::common_type_s(ctx, a, b);
}

Type _common_type_v(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_HAL_DISP(ctx, a, b);
  return mpc::common_type_v(ctx, a, b);
}

Value _cast_type_s(SPUContext* ctx, const Value& in, const Type& to) {
  SPU_TRACE_HAL_DISP(ctx, in, to);
  auto ret = mpc::cast_type_s(ctx, in, to);
  return ret;
}

Value _make_p(SPUContext* ctx, uint128_t init, const Shape& shape) {
  SPU_TRACE_HAL_DISP(ctx, init);
  auto res = mpc::make_p(ctx, init, shape);
  return res;
}

Value _rand_p(SPUContext* ctx, const Shape& shape) {
  SPU_TRACE_HAL_DISP(ctx, shape);
  auto rnd = mpc::rand_p(ctx, shape);
  return rnd;
}

Value _rand_s(SPUContext* ctx, const Shape& shape, DataType dtype) {
  SPU_TRACE_HAL_DISP(ctx, shape);
  auto rnd = mpc::rand_s(ctx, shape, dtype);
  return rnd;
}

Value _conv2d_ss(SPUContext* ctx, const Value& input, const Value& kernel,
                 const Strides& window_strides) {
  SPU_TRACE_HAL_DISP(ctx, input, kernel, window_strides);
  // FIXME(juhou): define conv2d_ss in api.h to capture this
  return dynDispatch(ctx, "conv2d_aa", input, kernel, window_strides[0],
                     window_strides[1]);
}

Value _trunc_p(SPUContext* ctx, const Value& in, size_t bits, SignType sign) {
  SPU_TRACE_HAL_DISP(ctx, in, bits, sign);
  return mpc::trunc_p(ctx, in, bits, sign);
}

Value _trunc_s(SPUContext* ctx, const Value& in, size_t bits, SignType sign) {
  SPU_TRACE_HAL_DISP(ctx, in, bits, sign);
  return mpc::trunc_s(ctx, in, bits, sign);
}

Value _trunc_v(SPUContext* ctx, const Value& in, size_t bits, SignType sign) {
  SPU_TRACE_HAL_DISP(ctx, in, bits, sign);
  return mpc::trunc_v(ctx, in, bits, sign);
}

std::optional<Value> _oramonehot_ss(SPUContext* ctx, const Value& x,
                                    int64_t db_size) {
  SPU_TRACE_HAL_DISP(ctx, x, db_size);
  return mpc::oram_onehot_ss(ctx, x, db_size);
}

std::optional<Value> _oramonehot_sp(SPUContext* ctx, const Value& x,
                                    int64_t db_size) {
  SPU_TRACE_HAL_DISP(ctx, x, db_size);
  return mpc::oram_onehot_sp(ctx, x, db_size);
}

Value _oramread_ss(SPUContext* ctx, const Value& x, const Value& y,
                   int64_t offset) {
  SPU_TRACE_HAL_DISP(ctx, x, y, offset);
  return mpc::oram_read_ss(ctx, x, y, offset);
}

Value _oramread_sp(SPUContext* ctx, const Value& x, const Value& y,
                   int64_t offset) {
  SPU_TRACE_HAL_DISP(ctx, x, y, offset);
  return mpc::oram_read_sp(ctx, x, y, offset);
}

// p<->s
MAP_UNARY_OP(p2s)
MAP_UNARY_OP(s2p)

// p<->v
MAP_UNARY_OP(v2p)
Value _p2v(SPUContext* ctx, const Value& in, int owner) {
  SPU_TRACE_HAL_DISP(ctx, in, owner);
  return mpc::p2v(ctx, in, owner);
}

// s<->v
MAP_UNARY_OP(v2s)
Value _s2v(SPUContext* ctx, const Value& in, int owner) {
  SPU_TRACE_HAL_DISP(ctx, in, owner);
  return mpc::s2v(ctx, in, owner);
}

// Not family
MAP_UNARY_OP(not_p)
MAP_UNARY_OP(not_s)
MAP_UNARY_OP(not_v)
// Negate family
MAP_UNARY_OP(negate_p)
MAP_UNARY_OP(negate_s)
MAP_UNARY_OP(negate_v)
// Msb family
MAP_UNARY_OP(msb_p)
MAP_UNARY_OP(msb_s)
MAP_UNARY_OP(msb_v)
// lshift family
MAP_SHIFT_OP(lshift_p)
MAP_SHIFT_OP(lshift_s)
MAP_SHIFT_OP(lshift_v)
// rshift family
MAP_SHIFT_OP(rshift_p)
MAP_SHIFT_OP(rshift_s)
MAP_SHIFT_OP(rshift_v)
// arshift family
MAP_SHIFT_OP(arshift_p)
MAP_SHIFT_OP(arshift_s)
MAP_SHIFT_OP(arshift_v)
// bitrev family
MAP_BITREV_OP(bitrev_p)
MAP_BITREV_OP(bitrev_s)
MAP_BITREV_OP(bitrev_v)
// Add family
MAP_BINARY_OP(add_pp)
MAP_BINARY_OP(add_sp)
MAP_BINARY_OP(add_ss)
MAP_BINARY_OP(add_sv)
MAP_BINARY_OP(add_vp)
MAP_BINARY_OP(add_vv)
// Mul family
MAP_BINARY_OP(mul_pp)
MAP_BINARY_OP(mul_sp)
MAP_BINARY_OP(mul_ss)
MAP_BINARY_OP(mul_sv)
MAP_BINARY_OP(mul_vp)
MAP_BINARY_OP(mul_vv)
MAP_UNARY_OP(square_p)
MAP_UNARY_OP(square_s)
MAP_UNARY_OP(square_v)
// And family
MAP_BINARY_OP(and_pp)
MAP_BINARY_OP(and_sp)
MAP_BINARY_OP(and_ss)
MAP_BINARY_OP(and_sv)
MAP_BINARY_OP(and_vp)
MAP_BINARY_OP(and_vv)
// Xor family
MAP_BINARY_OP(xor_pp)
MAP_BINARY_OP(xor_sp)
MAP_BINARY_OP(xor_ss)
MAP_BINARY_OP(xor_sv)
MAP_BINARY_OP(xor_vp)
MAP_BINARY_OP(xor_vv)
// mmul family
MAP_MMUL_OP(mmul_pp)
MAP_MMUL_OP(mmul_sp)
MAP_MMUL_OP(mmul_ss)
MAP_MMUL_OP(mmul_sv)
MAP_MMUL_OP(mmul_vp)
MAP_MMUL_OP(mmul_vv)

#define MAP_OPTIONAL_BINARY_OP(NAME)                                  \
  std::optional<Value> _##NAME(SPUContext* ctx, const Value& x,       \
                               const Value& y) {                      \
    SPU_TRACE_HAL_DISP(ctx, x, y);                                    \
    SPU_ENFORCE(x.shape() == y.shape(), "shape mismatch: x={}, y={}", \
                x.shape(), y.shape());                                \
    auto ret = mpc::NAME(ctx, x, y);                                  \
    if (!ret.has_value()) {                                           \
      return std::nullopt;                                            \
    }                                                                 \
    return ret.value();                                               \
  }

MAP_OPTIONAL_BINARY_OP(equal_ss)
MAP_OPTIONAL_BINARY_OP(equal_sp)
MAP_BINARY_OP(equal_pp)

#define MAP_OPTIONAL_PERM_OP(NAME)                                    \
  Value _##NAME(SPUContext* ctx, const Value& x, const Value& y) {    \
    SPU_TRACE_HAL_LEAF(ctx, x, y);                                    \
    SPU_ENFORCE(x.shape() == y.shape(), "shape mismatch: x={}, y={}", \
                x.shape(), y.shape());                                \
    SPU_ENFORCE(x.shape().ndim() == 1, "x should be a 1-d tensor");   \
    auto ret = mpc::NAME(ctx, x, y);                                  \
    SPU_ENFORCE(ret.has_value(), "{} api not implemented", #NAME);    \
    return ret.value().setDtype(x.dtype());                           \
  }  // namespace spu::kernel::hal

MAP_OPTIONAL_PERM_OP(perm_ss);
MAP_OPTIONAL_PERM_OP(perm_sp);
MAP_OPTIONAL_PERM_OP(inv_perm_ss);
MAP_OPTIONAL_PERM_OP(inv_perm_sp);
MAP_OPTIONAL_PERM_OP(inv_perm_sv);

Value _rand_perm_s(SPUContext* ctx, const Shape& shape) {
  SPU_TRACE_HAL_DISP(ctx, shape);
  SPU_ENFORCE(shape.ndim() == 1, "shape should be 1-d");
  auto ret = mpc::rand_perm_s(ctx, shape);
  SPU_ENFORCE(ret.has_value(), "rand_perm_s api not implemented");
  return ret.value();
}

Value _broadcast(SPUContext* ctx, const Value& in, const Shape& to_shape,
                 const Axes& in_dims) {
  return mpc::broadcast(ctx, in, to_shape, in_dims).setDtype(in.dtype());
}

Value _reshape(SPUContext* ctx, const Value& in, const Shape& to_shape) {
  return mpc::reshape(ctx, in, to_shape).setDtype(in.dtype());
}

Value _extract_slice(SPUContext* ctx, const Value& in,
                     const Index& start_indices, const Index& end_indices,
                     const Strides& strides) {
  return mpc::extract_slice(ctx, in, start_indices, end_indices, strides)
      .setDtype(in.dtype());
}

Value _update_slice(SPUContext* ctx, const Value& in, const Value& update,
                    const Index& start_indices) {
  return mpc::update_slice(ctx, in, update, start_indices).setDtype(in.dtype());
}

Value _transpose(SPUContext* ctx, const Value& in, const Axes& permutation) {
  return mpc::transpose(ctx, in, permutation).setDtype(in.dtype());
}

Value _reverse(SPUContext* ctx, const Value& in, const Axes& dimensions) {
  return mpc::reverse(ctx, in, dimensions).setDtype(in.dtype());
}

Value _fill(SPUContext* ctx, const Value& in, const Shape& to_shape) {
  return mpc::fill(ctx, in, to_shape).setDtype(in.dtype());
}

Value _pad(SPUContext* ctx, const Value& in, const Value& padding_value,
           const Sizes& edge_padding_low, const Sizes& edge_padding_high,
           const Sizes& interior_padding) {
  return mpc::pad(ctx, in, padding_value, edge_padding_low, edge_padding_high,
                  interior_padding)
      .setDtype(in.dtype());
}

Value _concatenate(SPUContext* ctx, const std::vector<Value>& values,
                   int64_t axis) {
  return mpc::concatenate(ctx, values, axis).setDtype(values.front().dtype());
}

Value _gen_inv_perm_p(SPUContext* ctx, const Value& in, bool is_ascending) {
  SPU_TRACE_HAL_LEAF(ctx, in, is_ascending);
  SPU_ENFORCE(in.shape().ndim() == 1, "input should be 1-d");
  return dynDispatch(ctx, "gen_inv_perm_p", in, is_ascending);
}

Value _gen_inv_perm_v(SPUContext* ctx, const Value& in, bool is_ascending) {
  SPU_TRACE_HAL_LEAF(ctx, in, is_ascending);
  SPU_ENFORCE(in.shape().ndim() == 1, "input should be 1-d");
  return dynDispatch(ctx, "gen_inv_perm_v", in, is_ascending);
}

Value _merge_keys_p(SPUContext* ctx, absl::Span<Value const> inputs,
                    bool is_ascending) {
  SPU_TRACE_HAL_LEAF(ctx, inputs.size(), inputs[0].shape(), is_ascending);
  std::vector<Value> in(inputs.begin(), inputs.end());
  return dynDispatch(ctx, "merge_keys_p", in, is_ascending);
}

Value _merge_keys_v(SPUContext* ctx, absl::Span<Value const> inputs,
                    bool is_ascending) {
  SPU_TRACE_HAL_LEAF(ctx, inputs.size(), inputs[0].shape(), is_ascending);
  std::vector<Value> in(inputs.begin(), inputs.end());
  return dynDispatch(ctx, "merge_keys_v", in, is_ascending);
}

Value _group_mark_p(SPUContext* ctx, absl::Span<Value const> inputs,
                    bool end_group_mark) {
  SPU_TRACE_HAL_LEAF(ctx, inputs.size(), inputs[0].shape(), end_group_mark);
  std::vector<Value> in(inputs.begin(), inputs.end());
  return dynDispatch(ctx, "group_mark_p", in, end_group_mark);
}

Value _group_mark_v(SPUContext* ctx, absl::Span<Value const> inputs,
                    bool end_group_mark) {
  SPU_TRACE_HAL_LEAF(ctx, inputs.size(), inputs[0].shape(), end_group_mark);
  std::vector<Value> in(inputs.begin(), inputs.end());
  return dynDispatch(ctx, "group_mark_v", in, end_group_mark);
}

#define MAP_PERM_OP(NAME)                                             \
  Value _##NAME(SPUContext* ctx, const Value& x, const Value& y) {    \
    SPU_TRACE_HAL_LEAF(ctx, x, y);                                    \
    SPU_ENFORCE(x.shape() == y.shape(), "shape mismatch: x={}, y={}", \
                x.shape(), y.shape());                                \
    SPU_ENFORCE(x.shape().ndim() == 1, "x should be a 1-d tensor");   \
    auto ret = mpc::NAME(ctx, x, y);                                  \
    return ret.setDtype(x.dtype());                                   \
  }

MAP_PERM_OP(inv_perm_pp);
MAP_PERM_OP(inv_perm_vv);
MAP_PERM_OP(perm_pp);
MAP_PERM_OP(perm_vv);

}  // namespace spu::kernel::hal
