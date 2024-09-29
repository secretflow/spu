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

#pragma once

#include "libspu/core/context.h"
#include "libspu/core/memref.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hal {

/// Debug print a value, this action may reveal secret.
// @param v, value to print
template <typename T>
std::string dbg_print(SPUContext* ctx, const MemRef& v) {
  if (v.isPublic()) {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, int128_t> || std::is_same_v<T, uint128_t>) {
      std::vector<T> ret = dump_public_as_vec<T>(ctx, v, ctx->getFxpBits());
      ss << fmt::format("{}", fmt::join(ret, ",")) << std::endl;
    } else {
      auto pt = dump_public_as<T>(ctx, v, ctx->getFxpBits());
      ss << pt << std::endl;
    }
    return fmt::format("dbg_print {}", ss.str());
  } else if (v.isSecret() || v.isPrivate()) {
    return dbg_print<T>(ctx, reveal(ctx, v));
  } else {
    SPU_THROW("unsupport vtype={}", v.vtype());
  }
}

}  // namespace spu::kernel::hal
