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
#include <optional>

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hlo {

size_t extractShiftBits(HalContext *ctx, const spu::Value &v) {
  YACL_ENFORCE(v.isInt());
  const auto arr = hal::dump_public(ctx, v);
  return DISPATCH_ALL_PT_TYPES(arr.eltype().as<PtTy>()->pt_type(), "", [&] {
    return static_cast<size_t>(arr.at<ScalarT>({}));
  });
}

template <typename Fn>
spu::Value shift_imp(HalContext *ctx, const spu::Value &lhs,
                     const spu::Value &rhs, const Fn &f) {
  YACL_ENFORCE(rhs.isPublic(), "shift bit value needs to be a public");
  YACL_ENFORCE(rhs.shape() == lhs.shape());
  std::vector<int64_t> indicies(lhs.shape().size(), 0);
  // Depend on protocol, shift result might be different, AShr vs BShr.
  // So delay the preallocation
  // FIXME: maybe we can do something better?
  std::vector<spu::Value> elements(lhs.numel());
  size_t idx = 0;
  do {
    auto bits = extractShiftBits(ctx, rhs.getElementAt(indicies));
    const auto lhs_el = lhs.getElementAt(indicies);
    elements[idx++] = f(ctx, lhs_el, bits);
  } while (bumpIndices<int64_t>(lhs.shape(), absl::MakeSpan(indicies)));

  // Compute common type
  auto common_type = elements.front().storage_type();
  for (size_t idx = 1; idx < elements.size(); ++idx) {
    common_type =
        hal::_common_type(ctx, common_type, elements[idx].storage_type());
  }

  spu::Value result({common_type, lhs.shape()}, lhs.dtype());
  // reset indicies
  std::fill(indicies.begin(), indicies.end(), 0);
  idx = 0;
  do {
    auto ret_el = hal::stype_cast(ctx, elements[idx++], common_type);
    result.copyElementFrom(ret_el, {}, indicies);
  } while (bumpIndices<int64_t>(lhs.shape(), absl::MakeSpan(indicies)));

  return result;
}

spu::Value Lshift(HalContext *ctx, const spu::Value &operand,
                  const spu::Value &bits_to_shift) {
  return shift_imp(ctx, operand, bits_to_shift, hal::left_shift);
}

spu::Value ARshift(HalContext *ctx, const spu::Value &operand,
                   const spu::Value &bits_to_shift) {
  return shift_imp(ctx, operand, bits_to_shift, hal::right_shift_arithmetic);
}

spu::Value Rshift(HalContext *ctx, const spu::Value &operand,
                  const spu::Value &bits_to_shift) {
  return shift_imp(ctx, operand, bits_to_shift, hal::right_shift_logical);
}

}  // namespace spu::kernel::hlo
