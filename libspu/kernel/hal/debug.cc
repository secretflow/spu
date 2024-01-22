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

#include "libspu/kernel/hal/debug.h"

#include "spdlog/spdlog.h"

#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hal {

void dbg_print(SPUContext* ctx, const Value& v) {
  if (v.isPublic()) {
    std::stringstream ss;
    if (v.isFxp()) {
      auto pt = dump_public_as<float>(ctx, v);
      ss << pt << std::endl;
    } else if (v.isInt()) {
      auto pt = dump_public_as<int64_t>(ctx, v);
      ss << pt << std::endl;
    } else {
      SPU_THROW("unsupport dtype={}", v.dtype());
    }
    if ((ctx->lctx() && ctx->lctx()->Rank() == 0) || ctx->lctx() == nullptr) {
      SPDLOG_INFO("dbg_print {}", ss.str());
    }
  } else if (v.isSecret() || v.isPrivate()) {
    dbg_print(ctx, reveal(ctx, v));
  } else {
    SPU_THROW("unsupport vtype={}", v.vtype());
  }
}

}  // namespace spu::kernel::hal
