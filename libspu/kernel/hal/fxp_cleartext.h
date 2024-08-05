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
#include "libspu/core/value.h"

// !!please read [README.md] for api naming conventions.
namespace spu::kernel::hal {

// This module decode public fxp back to flp and use std math function for
// evaluation.

Value f_reciprocal_p(SPUContext* ctx, const Value& in);

Value f_log_p(SPUContext* ctx, const Value& in);

Value f_exp_p(SPUContext* ctx, const Value& in);

Value f_div_p(SPUContext* ctx, const Value& x, const Value& y);

}  // namespace spu::kernel::hal
