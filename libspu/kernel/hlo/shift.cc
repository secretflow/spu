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

#include "libspu/kernel/hlo/shift.h"

#include <algorithm>
#include <cstddef>

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

template <typename Fn>
spu::Value shift_impl_p(SPUContext *ctx, const spu::Value &lhs,
                        const spu::Value &rhs, const Fn &f) {
  auto shift_bits = hal::dump_public_as<int8_t>(ctx, rhs);
  if (std::all_of(rhs.strides().begin(), rhs.strides().end(),
                  [](int64_t s) { return s == 0; })) {
    // rhs is a splat
    return f(ctx, lhs, shift_bits[0]);
  }

  // Not a splat...
  spu::Value ret =
      hal::constant(ctx, static_cast<uint8_t>(0), lhs.dtype(), lhs.shape());
  auto dtype_size = getWidth(lhs.dtype());
  for (size_t bits = 0; bits < dtype_size; ++bits) {
    if (std::none_of(shift_bits.begin(), shift_bits.end(), [&bits](int8_t b) {
          return b == static_cast<int8_t>(bits);
        })) {
      continue;
    }
    auto current_bits = hal::constant(ctx, static_cast<uint8_t>(bits),
                                      rhs.dtype(), rhs.shape());
    auto mask = hal::equal(ctx, rhs, current_bits);
    auto shifted = f(ctx, lhs, bits);
    ret = hal::add(ctx, ret, hal::mul(ctx, mask, shifted));
  }

  return ret;
}

template <typename Fn>
spu::Value shift_impl_s(SPUContext *ctx, const spu::Value &lhs,
                        const spu::Value &rhs, const Fn &f) {
  spu::Value ret =
      hal::constant(ctx, static_cast<uint8_t>(0), lhs.dtype(), lhs.shape());
  auto dtype_size = getWidth(lhs.dtype());
  for (size_t bits = 0; bits < dtype_size; ++bits) {
    auto current_bits = hal::constant(ctx, static_cast<uint8_t>(bits),
                                      rhs.dtype(), rhs.shape());
    auto mask = hal::equal(ctx, rhs, current_bits);
    auto shifted = f(ctx, lhs, bits);
    ret = hal::add(ctx, ret, hal::mul(ctx, mask, shifted));
  }

  return ret;
}

template <typename Fn>
spu::Value shift_impl(SPUContext *ctx, const spu::Value &lhs,
                      const spu::Value &rhs, const Fn &f) {
  SPU_ENFORCE(rhs.shape() == lhs.shape());

  if (rhs.isPublic()) {
    return shift_impl_p(ctx, lhs, rhs, f);
  }

  return shift_impl_s(ctx, lhs, rhs, f);
}

spu::Value Lshift(SPUContext *ctx, const spu::Value &operand,
                  const spu::Value &bits_to_shift) {
  return shift_impl(ctx, operand, bits_to_shift, hal::left_shift);
}

spu::Value ARshift(SPUContext *ctx, const spu::Value &operand,
                   const spu::Value &bits_to_shift) {
  return shift_impl(ctx, operand, bits_to_shift, hal::right_shift_arithmetic);
}

spu::Value Rshift(SPUContext *ctx, const spu::Value &operand,
                  const spu::Value &bits_to_shift) {
  return shift_impl(ctx, operand, bits_to_shift, hal::right_shift_logical);
}

}  // namespace spu::kernel::hlo
