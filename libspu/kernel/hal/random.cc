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

#include "libspu/core/prelude.h"
#include "libspu/core/xt_helper.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"

//
#include "libspu/mpc/common/prg_state.h"  // TODO: bad reference.

namespace spu::kernel::hal {
namespace {

uint64_t genPublicRandSeed(SPUContext* sctx) {
  auto* prg_state = sctx->prot()->getState<mpc::PrgState>();
  uint64_t seed;
  prg_state->fillPubl(absl::MakeSpan(&seed, 1));
  return seed;
}

}  // namespace

Value rng_uniform(SPUContext* ctx, const Value& lo, const Value& hi,
                  absl::Span<const int64_t> to_shape) {
  SPU_TRACE_HAL_LEAF(ctx, lo, hi, to_shape);
  SPU_ENFORCE(lo.isPublic() && hi.isPublic());
  SPU_ENFORCE(lo.numel() == 1 && hi.numel() == 1);

  const auto f_lo = getScalarValue<float>(ctx, lo);
  const auto f_hi = getScalarValue<float>(ctx, hi);

  // TODO: support more random generator.
  std::mt19937 gen(genPublicRandSeed(ctx));
  std::uniform_real_distribution<> dist(f_lo, f_hi);

  std::vector<float> buffer(calcNumel(to_shape));
  for (float& ele : buffer) {
    ele = dist(gen);
  }

  return constant(ctx, buffer, lo.dtype());
}

Value random(SPUContext* ctx, Visibility vis, DataType dtype,
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
