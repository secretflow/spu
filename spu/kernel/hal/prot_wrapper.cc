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

#include "spu/kernel/hal/prot_wrapper.h"

#include <cstddef>
#include <tuple>
#include <vector>

#include "yasl/base/exception.h"

#include "spu/core/array_ref.h"
#include "spu/core/ndarray_ref.h"
#include "spu/core/shape_util.h"
#include "spu/core/type_util.h"
#include "spu/mpc/api.h"

namespace spu::kernel::hal {
namespace {

Value unflattenValue(const ArrayRef& arr, std::vector<int64_t> shape) {
  // The underline MPC engine does not take care of dtype information, so we
  // should set it as INVALID and let the upper layer to handle it.
  return Value(unflatten(arr, shape), DT_INVALID);
}

ArrayRef flattenValue(const Value& v) {
  // TODO: optimization point.
  // we can set number of valids bits (according to dtype) to inform lower layer
  // for optimization. Currently dtype information is not used.
  return flatten(v.data());
}

std::tuple<int64_t, int64_t, int64_t> deduceMmulArgs(
    const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  YASL_ENFORCE(!lhs.empty() && lhs.size() <= 2);
  YASL_ENFORCE(!rhs.empty() && rhs.size() <= 2);

  if (lhs.size() == 1 && rhs.size() == 1) {
    YASL_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, 1, rhs[0]);
  }
  if (lhs.size() == 1 && rhs.size() == 2) {
    YASL_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, rhs[1], rhs[0]);
  }
  if (lhs.size() == 2 && rhs.size() == 1) {
    YASL_ENFORCE(lhs[1] == rhs[0]);
    return std::make_tuple(lhs[0], 1, rhs[0]);
  }
  YASL_ENFORCE(lhs[1] == rhs[0]);
  return std::make_tuple(lhs[0], rhs[1], rhs[0]);
}

}  // namespace

#define MAP_UNARY_OP(NAME)                               \
  Value _##NAME(HalContext* ctx, const Value& in) {      \
    SPU_TRACE_HAL_DISP(ctx, in);                         \
    auto ret = mpc::NAME(ctx->prot(), flattenValue(in)); \
    return unflattenValue(ret, in.shape());              \
  }

#define MAP_SHIFT_OP(NAME)                                       \
  Value _##NAME(HalContext* ctx, const Value& in, size_t bits) { \
    SPU_TRACE_HAL_DISP(ctx, in, bits);                           \
    auto ret = mpc::NAME(ctx->prot(), flattenValue(in), bits);   \
    return unflattenValue(ret, in.shape());                      \
  }

#define MAP_BITREV_OP(NAME)                                                   \
  Value _##NAME(HalContext* ctx, const Value& in, size_t start, size_t end) { \
    SPU_TRACE_HAL_DISP(ctx, in, start, end);                                  \
    auto ret = mpc::NAME(ctx->prot(), flattenValue(in), start, end);          \
    return unflattenValue(ret, in.shape());                                   \
  }

#define MAP_BINARY_OP(NAME)                                              \
  Value _##NAME(HalContext* ctx, const Value& x, const Value& y) {       \
    SPU_TRACE_HAL_DISP(ctx, x, y);                                       \
    YASL_ENFORCE(x.shape() == y.shape(), "shape mismatch: x={}, y={}",   \
                 x.shape(), y.shape());                                  \
    auto ret = mpc::NAME(ctx->prot(), flattenValue(x), flattenValue(y)); \
    return unflattenValue(ret, x.shape());                               \
  }

#define MAP_MMUL_OP(NAME)                                                  \
  Value _##NAME(HalContext* ctx, const Value& x, const Value& y) {         \
    SPU_TRACE_HAL_DISP(ctx, x, y);                                         \
    auto [m, n, k] = deduceMmulArgs(x.shape(), y.shape());                 \
    auto ret =                                                             \
        mpc::NAME(ctx->prot(), flattenValue(x), flattenValue(y), m, n, k); \
    return unflattenValue(ret, {m, n});                                    \
  }

Type _common_type_s(HalContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_HAL_DISP(ctx, a, b);
  return mpc::common_type_s(ctx->prot(), a, b);
}

Value _cast_type_s(HalContext* ctx, const Value& in, const Type& to) {
  SPU_TRACE_HAL_DISP(ctx, in, to);
  auto ret = mpc::cast_type_s(ctx->prot(), flattenValue(in), to);
  return unflattenValue(ret, in.shape());
}

MAP_UNARY_OP(p2s)
MAP_UNARY_OP(s2p)
MAP_UNARY_OP(not_p)
MAP_UNARY_OP(not_s)
MAP_UNARY_OP(msb_p)
MAP_UNARY_OP(msb_s)
MAP_UNARY_OP(eqz_p)
MAP_UNARY_OP(eqz_s)
MAP_SHIFT_OP(lshift_p)
MAP_SHIFT_OP(lshift_s)
MAP_SHIFT_OP(rshift_p)
MAP_SHIFT_OP(rshift_s)
MAP_SHIFT_OP(arshift_p)
MAP_SHIFT_OP(arshift_s)
MAP_SHIFT_OP(truncpr_s)
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

}  // namespace spu::kernel::hal
