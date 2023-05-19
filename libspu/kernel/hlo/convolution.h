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

#include "libspu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

struct ConvolutionConfig {
  int64_t featureGroupCount;
  int64_t batchGroupCount;
  absl::Span<const int64_t> window_strides;
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

// This is a port of hlo evaluator's HandleConvolutionWithLiterals, which can
// handle general convolution. See
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_evaluator_typed_visitor.h
spu::Value Convolution(SPUContext *ctx, const spu::Value &lhs,
                       const spu::Value &rhs, const ConvolutionConfig &config,
                       absl::Span<const int64_t> result_shape);

// This is an optimized conv2D with im2col
spu::Value Convolution2D(SPUContext *ctx, spu::Value input,
                         const spu::Value &kernel,
                         const ConvolutionConfig &config,
                         absl::Span<const int64_t> result_shape);

}  // namespace spu::kernel::hlo
