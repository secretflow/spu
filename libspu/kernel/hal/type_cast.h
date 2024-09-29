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

#include "libspu/core/memref.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

/// cast vtype
// @param in, the input value
// @param to_vis, destination vtype
MemRef vtype_cast(SPUContext* ctx, const MemRef& in, Visibility to_vis);

/// cast public to secret
// @param in, the input value
MemRef seal(SPUContext* ctx, const MemRef& x);

/// reveal a secret
// @param in, the input value
MemRef reveal(SPUContext* ctx, const MemRef& x);

}  // namespace spu::kernel::hal
