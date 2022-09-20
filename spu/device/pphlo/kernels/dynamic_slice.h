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

hal::Value DynamicUpdateSlice(HalContext *ctx, const hal::Value &operand,
                              const hal::Value &update,
                              absl::Span<const hal::Value> start_indicies);

hal::Value DynamicSlice(HalContext *ctx, const hal::Value &operand,
                        absl::Span<const int64_t> slice_size,
                        absl::Span<const hal::Value> start_indicies);

} // namespace spu::device::pphlo::kernel
