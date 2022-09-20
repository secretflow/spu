// Copyright 2022 Ant Group Co., Ltd.
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

#include "spu/device/pphlo/kernels/utils.h"

namespace spu::device::pphlo::kernel {

hal::Value Cast(HalContext *ctx, const hal::Value &in, Visibility dst_vtype,
                DataType dst_dtype);

hal::Value Bitcast(HalContext *ctx, const hal::Value &in, DataType dst_dtype,
                   size_t elsize);

hal::Value Reveal(HalContext *ctx, const hal::Value &in);

hal::Value Seal(HalContext *ctx, const hal::Value &in);

} // namespace spu::device::pphlo::kernel
