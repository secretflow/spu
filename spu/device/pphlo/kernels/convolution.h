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

struct ConvolutionConfig {
  int64_t featureGroupCount;
  int64_t batchGroupCount;
  absl::Span<const int64_t> window_strides;
  absl::Span<const int64_t> padding;
  absl::Span<const int64_t> lhs_dilation;
  absl::Span<const int64_t> rhs_dilation;
  int64_t inputBatchDimension;
  int64_t inputFeatureDimension;
  absl::Span<const int64_t> inputSpatialDimensions;
  int64_t kernelInputFeatureDimension;
  int64_t kernelOutputFeatureDimension;
  absl::Span<const int64_t> kernelSpatialDimensions;
  int64_t outputBatchDimension;
  int64_t outputFeatureDimension;
  absl::Span<const int64_t> outputSpatialDimensions;
};

// This is a port of hlo evoluator's HandleConvolutionWithLiterals, which can
// handle general convolution. See
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_evaluator_typed_visitor.h
hal::Value Convolution(HalContext *ctx, const hal::Value &lhs,
                       const hal::Value &rhs, const ConvolutionConfig &config,
                       absl::Span<const int64_t> result_shape);

// This is an optimized conv2D with im2col
hal::Value Convolution2D(HalContext *ctx, hal::Value input, hal::Value kernel,
                         const ConvolutionConfig &config);

} // namespace spu::device::pphlo::kernel
