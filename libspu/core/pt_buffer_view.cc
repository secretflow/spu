// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/core/pt_buffer_view.h"

#include "libspu/core/shape.h"
#include "libspu/core/type_util.h"

namespace spu {

namespace detail {

bool isCompact(const Strides& stride, const Shape& shape) {
  if (shape.numel() < 2) {
    return true;
  }
  return stride == makeCompactStrides(shape);
}

}  // namespace detail

std::ostream& operator<<(std::ostream& out, PtBufferView v) {
  out << fmt::format("PtBufferView<{},{}x{},{}>", v.ptr,
                     fmt::join(v.shape, "x"), v.pt_type,
                     fmt::join(v.strides, "x"));
  return out;
}

NdArrayRef convertToNdArray(PtBufferView bv) {
  if (bv.isBitSet()) {
    SPU_ENFORCE(bv.isCompact() && bv.pt_type == PT_I1);
    auto out = NdArrayRef(I1, bv.shape);
    auto* out_ptr = out.data<bool>();
    auto num_bits = bv.shape.numel();
    for (int64_t idx = 0; idx < num_bits; ++idx) {
      out_ptr[idx] = bv.getBit(idx);
    }
    return out;
  }
  const auto type = makePtType(bv.pt_type);
  auto out = NdArrayRef(type, bv.shape);
  return DISPATCH_ALL_PT_TYPES(bv.pt_type, "pt_type", [&]() {
    using T = ScalarT;
    if (bv.shape.numel() > 0) {
      auto* out_ptr = out.data<T>();

      Index indices(bv.shape.size(), 0);
      if (bv.isCompact()) {
        std::memcpy(out_ptr, &bv.get<T>(indices), sizeof(T) * bv.shape.numel());
      } else {
        do {
          *out_ptr = bv.get<T>(indices);
          out_ptr += 1;
        } while (bumpIndices(bv.shape, absl::MakeSpan(indices)));
      }
    }
    return out;
  });
}

}  // namespace spu
