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

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hlo {

struct ConvolutionConfig {
  Strides window_strides;
  int64_t inputBatchDimension;
  int64_t inputFeatureDimension;
  Axes inputSpatialDimensions;
  int64_t kernelInputFeatureDimension;
  int64_t kernelOutputFeatureDimension;
  Axes kernelSpatialDimensions;
  int64_t outputBatchDimension;
  int64_t outputFeatureDimension;
  Axes outputSpatialDimensions;
};

// This is an optimized conv2D with im2col
spu::Value Convolution2D(SPUContext *ctx, const spu::Value &input,
                         const spu::Value &kernel,
                         const ConvolutionConfig &config,
                         const Shape &result_shape);

}  // namespace spu::kernel::hlo
