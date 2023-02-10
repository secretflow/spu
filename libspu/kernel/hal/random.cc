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

#include "libspu/kernel/hal/random.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"

#include "libspu/core/prelude.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/test_util.h"

namespace spu::kernel::hal {

Value rng_uniform(HalContext* ctx, const Value& a, const Value& b,
                  absl::Span<const int64_t> to_shape) {
  SPU_TRACE_HAL_LEAF(ctx, a, b, to_shape);
  SPU_ENFORCE(a.isPublic() && b.isPublic());
  SPU_ENFORCE(a.dtype() == b.dtype());
  // FIXME: This is a hacky ref impl, fill a real proper impl later.
  if (a.isFxp()) {
    auto pa = test::dump_public_as<float>(ctx, a);
    auto pb = test::dump_public_as<float>(ctx, b);
    xt::xarray<float> randv =
        xt::random::rand(to_shape, pa[0], pb[0], ctx->rand_engine());
    return constant(ctx, randv);
  }

  SPU_ENFORCE(a.isInt());
  auto pa = test::dump_public_as<int>(ctx, a);
  auto pb = test::dump_public_as<int>(ctx, b);
  xt::xarray<int> randv =
      xt::random::randint(to_shape, pa[0], pb[0], ctx->rand_engine());
  return constant(ctx, randv);
}

Value random(HalContext* ctx, Visibility vis, DataType dtype,
             absl::Span<const int64_t> shape) {
  Value ret;
  if (vis == VIS_PUBLIC) {
    ret = _rand_p(ctx, shape).setDtype(dtype);
  } else if (vis == VIS_SECRET) {
    ret = _rand_s(ctx, shape).setDtype(dtype);
  } else {
    SPU_THROW("Invalid visibility={}", vis);
  }

  return ret;
}

}  // namespace spu::kernel::hal
