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

#pragma once

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/core/xt_helper.h"

namespace spu::kernel::hal {

// Export a value to a buffer.
NdArrayRef dump_public(SPUContext* ctx, const Value& v);

// TODO: Find a more proper space.
template <typename T>
xt::xarray<T> dump_public_as(SPUContext* ctx, const Value& in) {
  auto arr = dump_public(ctx, in);

#define CASE(NAME, TYPE, _)                  \
  case NAME: {                               \
    return xt::cast<T>(xt_adapt<TYPE>(arr)); \
  }

  switch (arr.eltype().as<PtTy>()->pt_type()) {
    FOREACH_PT_TYPES(CASE)

    default:
      SPU_THROW("unexpected type={}", arr.eltype());
  }

#undef CASE
}

template <typename T>
T getScalarValue(SPUContext* ctx, const spu::Value& value) {
  SPU_ENFORCE(value.numel() == 1, "{} is not a scalar tensor.", value);
  SPU_ENFORCE(value.isPublic(), "{} is not a public value", value);

  const auto pvar = kernel::hal::dump_public_as<T>(ctx, value);
  return pvar.front();
}

}  // namespace spu::kernel::hal
