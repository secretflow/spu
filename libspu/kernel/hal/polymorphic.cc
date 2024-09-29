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

#include "libspu/kernel/hal/polymorphic.h"

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/hal/ring.h"  // for fast fxp x int
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"

// TODO: handle dtype promotion inside integer dtypes.
namespace spu::kernel::hal {

MemRef logical_not(SPUContext* ctx, const MemRef& in) {
  SPU_TRACE_HAL_LEAF(ctx, in);

  auto _k1 =
      _constant(ctx, 1, in.eltype().as<RingTy>()->semantic_type(), in.shape());

  // TODO: we should NOT dispatch according to AShr/BShr trait here.
  if (in.eltype().isa<BoolShare>()) {
    return _xor(ctx, in, _k1);
  } else {
    return _sub(ctx, _k1, in);
  }
}

MemRef not_equal(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return logical_not(ctx, _equal(ctx, x, y));
}

MemRef less_equal(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  // not (x > y)
  return logical_not(ctx, greater(ctx, x, y));
}

MemRef greater(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return _less(ctx, y, x);
}

MemRef greater_equal(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  // not (x < y)
  return logical_not(ctx, _less(ctx, x, y));
}

MemRef clamp(SPUContext* ctx, const MemRef& x, const MemRef& minv,
             const MemRef& maxv) {
  SPU_TRACE_HAL_DISP(ctx, x, minv, maxv);

  auto max = [](SPUContext* ctx, const MemRef& x, const MemRef& y) {
    return _mux(ctx, greater(ctx, x, y), x, y);
  };

  auto min = [](SPUContext* ctx, const MemRef& x, const MemRef& y) {
    return _mux(ctx, _less(ctx, x, y), x, y);
  };

  return min(ctx, max(ctx, minv, x), maxv);
}

MemRef f_floor(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  const int64_t fbits = ctx->getFxpBits();
  return _lshift(ctx, _arshift(ctx, x, {fbits}), {fbits});
}

spu::MemRef round_tne(SPUContext* ctx, const spu::MemRef& in) {
  // RNTE: Round to nearest, ties to even
  // let x^' = *****a.b##### be origin fxp number
  // x = *****a.bc ( c = reduce_or(#####) ), y = *****a
  // then ret = y + comp (comp = 0 or 1), where
  // 1) if b=0, then comp=0
  // 2) if b=1, c=1, then comp=1
  // 3) if b=1, c=0, a=1, then comp=1
  // 4) if b=1, c=0, a=0, then comp=0
  // so comp = b && (c || a)
  const auto fxp_bits = ctx->getFxpBits();
  const auto k1 = hal::_constant(
      ctx, 1U, in.eltype().as<RingTy>()->semantic_type(), in.shape());

  auto x_prime = hal::_prefer_b(ctx, in);
  auto y = hal::f_floor(ctx, x_prime);

  auto a = hal::_and(
      ctx, hal::_rshift(ctx, x_prime, {static_cast<int64_t>(fxp_bits)}), k1);
  auto b = hal::_and(
      ctx, hal::_rshift(ctx, x_prime, {static_cast<int64_t>(fxp_bits - 1)}),
      k1);

  std::vector<MemRef> cs;
  cs.reserve(fxp_bits - 1);
  for (size_t idx = 0; idx < fxp_bits - 1; idx++) {
    auto x_ = hal::_and(
        ctx, hal::_rshift(ctx, x_prime, {static_cast<int64_t>(idx)}), k1);
    cs.push_back(std::move(x_));
  }
  auto c = vreduce(cs.begin(), cs.end(), [&](const MemRef& a, const MemRef& b) {
    return hal::_or(ctx, a, b);
  });
  auto comp = hal::_and(ctx, b, hal::_or(ctx, c, a));
  // set nbits to improve b2a
  if (comp.eltype().isa<BoolShare>()) {
    const_cast<Type&>(comp.eltype()).as<BaseRingType>()->set_valid_bits(1);
  }

  // int -> fxp
  comp = hal::_lshift(ctx, comp, {static_cast<int64_t>(fxp_bits)});

  return hal::_add(ctx, y, comp);
}

std::optional<MemRef> oramonehot(SPUContext* ctx, const MemRef& x,
                                 int64_t db_size, bool db_is_secret) {
  auto ret = _oramonehot(ctx, x, db_size, db_is_secret);
  if (!ret.has_value()) {
    return std::nullopt;
  }
  return ret;
}

MemRef oramread(SPUContext* ctx, const MemRef& x, const MemRef& y,
                int64_t offset) {
  MemRef ret = _oramread(ctx, x, y, offset);
  return ret;
}

}  // namespace spu::kernel::hal
