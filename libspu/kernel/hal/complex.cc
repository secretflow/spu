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

#include "libspu/kernel/hal/complex.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hal {

MemRef real(SPUContext*, const MemRef& v) {
  SPU_ENFORCE(v.isComplex());
  auto new_strides = v.strides();
  for (auto& s : new_strides) {
    s *= 2;
  }
  return MemRef(v.buf(), v.eltype(), v.shape(), new_strides, v.offset());
}

MemRef imag(SPUContext* ctx, const MemRef& v) {
  if (v.isComplex()) {
    auto new_strides = v.strides();
    for (auto& s : new_strides) {
      s *= 2;
    }
    return MemRef(v.buf(), v.eltype(), v.shape(), new_strides,
                  v.offset() + v.eltype().size());
  } else {
    auto zeros = hal::zeros(ctx, PT_I64, v.shape());
    if (v.isSecret()) {
      return hal::seal(ctx, zeros);
    }
    return zeros;
  }
}

MemRef complex(SPUContext*, const MemRef& r, const MemRef& i) {
  SPU_ENFORCE(r.vtype() == i.vtype());
  SPU_ENFORCE(r.shape() == i.shape());
  SPU_ENFORCE(r.eltype() == i.eltype());
  SPU_ENFORCE(!r.isComplex() && !i.isComplex());
  auto buf = std::make_shared<yacl::Buffer>(r.numel() * r.elsize() * 2);
  Strides strides = makeCompactStrides(r.shape());
  for (auto& s : strides) {
    s *= 2;
  }
  MemRef real_encoded(buf, r.eltype(), r.shape(), strides, 0);
  MemRef imag_encoded(buf, r.eltype(), r.shape(), strides, r.elsize());

  for (int64_t j = 0; j < r.numel(); ++j) {
    std::memcpy((void*)&real_encoded.at(j), (void*)&r.at(j), r.elsize());
    std::memcpy((void*)&imag_encoded.at(j), (void*)&i.at(j), i.elsize());
  }

  return MemRef(buf, r.eltype(), r.shape(), true);
}

}  // namespace spu::kernel::hal
