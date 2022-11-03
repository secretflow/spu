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

#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

struct GatherConfig {
  absl::Span<const int64_t> sliceSizes;
  int64_t indexVectorDim;
  absl::Span<const int64_t> offsetDims;
  absl::Span<const int64_t> collapsedSliceDims;
  absl::Span<const int64_t> startIndexMap;
};

// This is ported from
// https://github.com/tensorflow/tensorflow/blob/bf4c6ad46dac1f7f69911e2bfc48e141a39b40af/tensorflow/compiler/xla/service/hlo_evaluator.cc#L1774
spu::Value Gather(HalContext *ctx, const spu::Value &operand,
                  const spu::Value &start_indicies, const GatherConfig &config,
                  absl::Span<const int64_t> result_shape);

}  // namespace spu::kernel::hlo
