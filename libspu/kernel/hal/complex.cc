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

Value real(SPUContext*, const Value& v) { return Value(v.data(), v.dtype()); }

Value imag(SPUContext* ctx, const Value& v) {
  if (v.isComplex()) {
    return Value(*v.imag(), v.dtype());  // NOLINT
  } else {
    auto zeros = hal::zeros(ctx, v.dtype(), v.shape());
    if (v.isSecret()) {
      return hal::seal(ctx, zeros);
    }
    return zeros;
  }
}

Value complex(SPUContext*, const Value& r, const Value& i) {
  SPU_ENFORCE(r.dtype() == r.dtype());
  SPU_ENFORCE(r.vtype() == r.vtype());
  SPU_ENFORCE(r.shape() == r.shape());
  SPU_ENFORCE(!r.isComplex() && !i.isComplex());

  return Value(r.data(), i.data(), r.dtype());
}
}  // namespace spu::kernel::hal
