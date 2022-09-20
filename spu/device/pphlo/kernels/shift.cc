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

#include "spu/device/pphlo/kernels/shift.h"

#include <optional>

#include "spu/hal/constants.h"
#include "spu/hal/polymorphic.h"

namespace spu::device::pphlo::kernel {

size_t extractShiftBits(HalContext *ctx, const hal::Value &v) {
  YASL_ENFORCE(v.isInt());
  const auto arr = hal::dump_public(ctx, v);
  return DISPATCH_ALL_PT_TYPES(arr.eltype().as<PtTy>()->pt_type(), "", [&] {
    return static_cast<size_t>(arr.at<ScalarT>({}));
  });
}

template <typename Fn>
hal::Value shift_imp(HalContext *ctx, const hal::Value &lhs,
                     const hal::Value &rhs, const Fn &f) {
  YASL_ENFORCE(rhs.isPublic(), "shift bit value needs to be a public");
  YASL_ENFORCE(rhs.shape() == lhs.shape());
  std::vector<int64_t> indicies(lhs.shape().size(), 0);
  // Depend on protocl, shift result might be different, AShr vs BShr.
  // So delay the preallocation
  // FIXME: maybe we can do something better?
  std::optional<hal::Value> result;
  do {
    auto bits = extractShiftBits(ctx, rhs.getElementAt(indicies));
    const auto lhs_el = lhs.getElementAt(indicies);
    auto ret_el = f(ctx, lhs_el, bits);
    if (!result.has_value()) {
      result = hal::Value({ret_el.storage_type(), lhs.shape()}, lhs.dtype());
    }
    result->copyElementFrom(ret_el, {}, indicies);
  } while (bumpIndices<int64_t>(lhs.shape(), absl::MakeSpan(indicies)));

  return result.value();
}

hal::Value Lshift(HalContext *ctx, const hal::Value &operand,
                  const hal::Value &bits_to_shift) {
  return shift_imp(ctx, operand, bits_to_shift, hal::left_shift);
}

hal::Value ARshift(HalContext *ctx, const hal::Value &operand,
                   const hal::Value &bits_to_shift) {
  return shift_imp(ctx, operand, bits_to_shift, hal::right_shift_arithmetic);
}

hal::Value Rshift(HalContext *ctx, const hal::Value &operand,
                  const hal::Value &bits_to_shift) {
  return shift_imp(ctx, operand, bits_to_shift, hal::right_shift_logical);
}

} // namespace spu::device::pphlo::kernel
