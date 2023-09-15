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

namespace spu {

std::ostream& operator<<(std::ostream& out, PtBufferView v) {
  out << fmt::format("PtBufferView<{},{}x{},{}>", v.ptr,
                     fmt::join(v.shape, "x"), v.pt_type,
                     fmt::join(v.strides, "x"));
  return out;
}

NdArrayRef convertToNdArray(PtBufferView bv) {
  const auto type = makePtType(bv.pt_type);
  auto out = NdArrayRef(type, bv.shape);

  if (bv.shape.numel() > 0) {
    auto* out_ptr = out.data<std::byte>();

    size_t elsize = SizeOf(bv.pt_type);

    Index indices(bv.shape.size(), 0);
    if (bv.isCompact()) {
      std::memcpy(out_ptr, bv.get(indices), elsize * bv.shape.numel());
    } else {
      do {
        std::memcpy(out_ptr, bv.get(indices), elsize);
        out_ptr += elsize;
      } while (bumpIndices(bv.shape, absl::MakeSpan(indices)));
    }
  }

  return out;
}

}  // namespace spu
