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
#include <tuple>
#include <vector>

#include "libspu/core/array_ref.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/shape_util.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/api.h"

namespace spu::kernel::hal {
namespace {

std::tuple<int64_t, int64_t, int64_t> deduceMmulArgs(
    const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  SPU_ENFORCE(!lhs.empty() && lhs.size() <= 2);
  SPU_ENFORCE(!rhs.empty() && rhs.size() <= 2);

  if (lhs.size() == 1 && rhs.size() == 1) {
    SPU_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, 1, rhs[0]);
  }
  if (lhs.size() == 1 && rhs.size() == 2) {
    SPU_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, rhs[1], rhs[0]);
  }
  if (lhs.size() == 2 && rhs.size() == 1) {
    SPU_ENFORCE(lhs[1] == rhs[0]);
    return std::make_tuple(lhs[0], 1, rhs[0]);
  }
  SPU_ENFORCE(lhs[1] == rhs[0]);
  return std::make_tuple(lhs[0], rhs[1], rhs[0]);
}

}  // namespace

#define MAP_UNARY_OP(NAME)                          \
  Value _##NAME(SPUContext* ctx, const Value& in) { \
    SPU_TRACE_HAL_DISP(ctx, in);                    \
    return mpc::NAME(ctx, in);                      \
  }

#define MAP_SHIFT_OP(NAME)                                       \
  Value _##NAME(SPUContext* ctx, const Value& in, size_t bits) { \
    SPU_TRACE_HAL_DISP(ctx, in, bits);                           \
    auto ret = mpc::NAME(ctx, in, bits);                         \
    return ret;                                                  \
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
    auto [m, n, k] = deduceMmulArgs(x.shape(), y.shape());         \
    auto ret = mpc::NAME(ctx, x, y, m, n, k);                      \
    return ret;                                                    \
  }

Type _common_type_s(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_HAL_DISP(ctx, a, b);
  return mpc::common_type_s(ctx, a, b);
}

Value _cast_type_s(SPUContext* ctx, const Value& in, const Type& to) {
  SPU_TRACE_HAL_DISP(ctx, in, to);
  auto ret = mpc::cast_type_s(ctx, in, to);
  return ret;
}

Value _make_p(SPUContext* ctx, uint128_t init,
              absl::Span<const int64_t> shape) {
  SPU_TRACE_HAL_DISP(ctx, init);
  auto res = mpc::make_p(ctx, init, Shape(shape));
  return res;
}

Value _rand_p(SPUContext* ctx, absl::Span<const int64_t> shape) {
  SPU_TRACE_HAL_DISP(ctx, shape);
  auto rnd = mpc::rand_p(ctx, Shape(shape));
  return rnd;
}

Value _rand_s(SPUContext* ctx, absl::Span<const int64_t> shape) {
  SPU_TRACE_HAL_DISP(ctx, shape);
  auto rnd = mpc::rand_s(ctx, Shape(shape));
  return rnd;
}

Value _conv2d_ss(SPUContext* ctx, Value input, const Value& kernel,
                 absl::Span<const int64_t> window_strides,
                 absl::Span<const int64_t> result_shape) {
  SPU_TRACE_HAL_DISP(ctx, input, kernel, window_strides, result_shape);
  SPU_ENFORCE_EQ(window_strides.size(), 2UL);
  size_t N = input.shape()[0];
  size_t C = input.shape()[3];

  size_t h = kernel.shape()[0];
  size_t w = kernel.shape()[1];
  size_t O = kernel.shape()[3];
  size_t stride_w = window_strides[0];
  size_t stride_h = window_strides[1];
  SPU_ENFORCE_EQ(result_shape[0], static_cast<int64_t>(N));
  SPU_ENFORCE_EQ(result_shape[3], static_cast<int64_t>(O));
  SPU_ENFORCE_EQ(kernel.shape()[2], static_cast<int64_t>(C));

  // ad-hoc optimization for strided conv2d when h=1
  std::vector<int64_t> strides = {1, 1, 1, 1};
  if (h == 1) {
    strides[1] = stride_h;
  }
  if (w == 1) {
    strides[2] = stride_w;
  }

  if (std::any_of(strides.begin(), strides.end(),
                  [](int64_t s) { return s > 1; })) {
    input = Value(input.data().slice({0, 0, 0, 0}, input.shape(), strides),
                  input.dtype());

    stride_h = 1;
    stride_w = 1;
  }

  size_t H = input.shape()[1];
  size_t W = input.shape()[2];
  // FIXME(juhou): define conv2d_ss in api.h to capture this
  return dynDispatch(ctx, "conv2d_aa", input, kernel, N, H, W, C, O, h, w,
                     stride_h, stride_w);
}

Value _trunc_p_with_sign(SPUContext* ctx, const Value& in, size_t bits,
                         bool /*dummy*/) {
  return _trunc_p(ctx, in, bits);
}

Value _trunc_s_with_sign(SPUContext* ctx, const Value& in, size_t bits,
                         bool is_positive) {
  if (ctx->config().protocol() == ProtocolKind::CHEETAH) {
    return dynDispatch(ctx, "trunc_a_with_sign", in, bits, is_positive);
  } else {
    return _trunc_s(ctx, in, bits);
  }
}

MAP_UNARY_OP(p2s)
MAP_UNARY_OP(s2p)
MAP_UNARY_OP(not_p)
MAP_UNARY_OP(not_s)
MAP_UNARY_OP(msb_p)
MAP_UNARY_OP(msb_s)
MAP_SHIFT_OP(lshift_p)
MAP_SHIFT_OP(lshift_s)
MAP_SHIFT_OP(rshift_p)
MAP_SHIFT_OP(rshift_s)
MAP_SHIFT_OP(arshift_p)
MAP_SHIFT_OP(arshift_s)
MAP_SHIFT_OP(trunc_p)
MAP_SHIFT_OP(trunc_s)
MAP_BITREV_OP(bitrev_p)
MAP_BITREV_OP(bitrev_s)
MAP_BINARY_OP(add_pp)
MAP_BINARY_OP(add_sp)
MAP_BINARY_OP(add_ss)
MAP_BINARY_OP(mul_pp)
MAP_BINARY_OP(mul_sp)
MAP_BINARY_OP(mul_ss)
MAP_BINARY_OP(and_pp)
MAP_BINARY_OP(and_sp)
MAP_BINARY_OP(and_ss)
MAP_BINARY_OP(xor_pp)
MAP_BINARY_OP(xor_sp)
MAP_BINARY_OP(xor_ss)
MAP_MMUL_OP(mmul_pp)
MAP_MMUL_OP(mmul_sp)
MAP_MMUL_OP(mmul_ss)

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

}  // namespace spu::kernel::hal
