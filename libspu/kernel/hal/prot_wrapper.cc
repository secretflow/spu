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

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/api.h"

namespace spu::kernel::hal {

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
    auto ret = mpc::NAME(ctx, x, y);                               \
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

Value _rand_s(SPUContext* ctx, const Shape& shape) {
  SPU_TRACE_HAL_DISP(ctx, shape);
  auto rnd = mpc::rand_s(ctx, shape);
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

std::vector<Value> _sort_s(SPUContext* ctx, absl::Span<Value const> x) {
  SPU_TRACE_HAL_DISP(ctx, x.size());
  // FIXME(jimi): formalize mpc sort api
  return dynDispatch<std::vector<Value>>(ctx, "sort_a", x);
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
