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

#include "libspu/kernel/hlo/convolution.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/utils.h"

namespace {

std::vector<int64_t> MakeDimMultipliers(absl::Span<const int64_t> shape) {
  std::vector<int64_t> v(shape.size());
  int64_t scale = 1;
  for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
    v[dim] = scale;
    scale *= shape[dim];
  }
  return v;
}

}  // namespace

namespace spu::kernel::hlo {

spu::Value Convolution(SPUContext *ctx, const spu::Value &lhs,
                       const spu::Value &rhs, const ConvolutionConfig &config,
                       absl::Span<const int64_t> result_shape) {
  const size_t num_spatial_dims = config.outputSpatialDimensions.size();
  SPU_ENFORCE(num_spatial_dims == config.inputSpatialDimensions.size());
  SPU_ENFORCE(num_spatial_dims == config.kernelSpatialDimensions.size());

  const auto &lhs_shape = lhs.shape();

  const auto &rhs_shape = rhs.shape();

  std::vector<int64_t> window_shape;
  for (auto i : config.kernelSpatialDimensions) {
    window_shape.push_back(rhs_shape[i]);
  }

  auto lhs_dim_multipliers = MakeDimMultipliers(lhs_shape);
  auto rhs_dim_multipliers = MakeDimMultipliers(rhs_shape);

  // Dimension number applicable for input (lhs).
  const int64_t input_batch_dim = config.inputBatchDimension;
  const int64_t input_z_dim = config.inputFeatureDimension;
  // Dimension number applicable for kernel (rhs).
  const int64_t kernel_input_z_dim = config.kernelInputFeatureDimension;
  const int64_t kernel_output_z_dim = config.kernelOutputFeatureDimension;
  // Dimension number applicable for output.
  const int64_t output_batch_dim = config.outputBatchDimension;
  const int64_t output_z_dim = config.outputFeatureDimension;

  const int64_t input_z_size = lhs_shape[input_z_dim];

  const int64_t input_batch_size = lhs_shape[input_batch_dim];

  const int64_t batch_group_size = input_batch_size / config.batchGroupCount;

  // The size of an input feature group.
  const int64_t input_feature_group_size =
      input_z_size / config.featureGroupCount;

  const int64_t output_z_size = rhs_shape[kernel_output_z_dim];
  // The output feature dimension is a concatenation of convolution results
  // from the different groups.
  const int64_t output_feature_group_size =
      output_z_size / config.featureGroupCount;

  // Start computing
  spu::Value ret;

  // Iterate on window
  std::vector<int64_t> window_index(config.kernelSpatialDimensions.size(), 0);

  do {
    spu::Value lhs_slice = hal::zeros(ctx, lhs.dtype(), result_shape);
    spu::Value rhs_slice = hal::zeros(ctx, rhs.dtype(), result_shape);

    if (lhs.isSecret()) {
      lhs_slice = hal::seal(ctx, lhs_slice);
    }
    if (rhs.isSecret()) {
      rhs_slice = hal::seal(ctx, rhs_slice);
    }

    forEachIndex(result_shape, [&](absl::Span<const int64_t> output_index) {
      // Calculate the group index to which the current output index
      // belongs.
      const int64_t feature_group_index =
          output_index[output_z_dim] / output_feature_group_size;

      const int64_t depthwise_multiplier =
          config.batchGroupCount > 1 ? output_z_size / input_batch_size : 1;
      const int64_t batch_group_index =
          output_index[output_z_dim] / depthwise_multiplier;

      // Find corresponding spatial dimension index for input (lhs).
      int64_t lhs_linear_spatial_index = 0;
      int64_t rhs_linear_spatial_index = 0;
      for (int64_t ki = 0; ki < static_cast<int64_t>(window_index.size());
           ++ki) {
        // Spatial dimension number for input (lhs) and output.
        const int64_t input_spatial_dim = config.inputSpatialDimensions[ki];
        const int64_t output_spatial_dim = config.outputSpatialDimensions[ki];

        // Calculate lhs (input) index without taking base dilation into
        // account.
        const int64_t lhs_spatial_index =
            output_index[output_spatial_dim] * config.window_strides[ki] +
            window_index[ki];

        // Skip if input index is not in bounds.
        if (lhs_spatial_index < 0 ||
            lhs_spatial_index >= lhs_shape[input_spatial_dim]) {
          return;
        }

        lhs_linear_spatial_index +=
            lhs_spatial_index * lhs_dim_multipliers[input_spatial_dim];
        rhs_linear_spatial_index +=
            window_index[ki] *
            rhs_dim_multipliers[config.kernelSpatialDimensions[ki]];
      }

      for (int64_t rhs_iz = 0; rhs_iz < input_feature_group_size; ++rhs_iz) {
        const int64_t iz =
            feature_group_index * input_feature_group_size + rhs_iz;

        int64_t lhs_linear_index = lhs_linear_spatial_index;
        lhs_linear_index += output_index[output_batch_dim] *
                            lhs_dim_multipliers[input_batch_dim];

        // We are scraping only the diagonal elements in the resultant
        // convolution output when batch_group_count is greater than 1,
        // where 1 is the default. No scraping is done in that case.
        // This approach works out automatically for 'groups' in batches
        // with group_size > 1, because we already descend down the batch
        // dimension for the 'output_batch_dim' above.
        lhs_linear_index +=
            ((batch_group_index * batch_group_size) % input_batch_size) *
            lhs_dim_multipliers[input_batch_dim];

        lhs_linear_index += iz * lhs_dim_multipliers[input_z_dim];
        int64_t rhs_linear_index = rhs_linear_spatial_index;

        rhs_linear_index += output_index[output_z_dim] *
                            rhs_dim_multipliers[kernel_output_z_dim];
        rhs_linear_index += rhs_iz * rhs_dim_multipliers[kernel_input_z_dim];

        // TODO: anti-pattern, do not use .data(), use ops instead.
        lhs_slice.data().update_slice(lhs.data().slice_scalar_at(unflattenIndex(
                                          lhs_linear_index, lhs.shape())),
                                      output_index);
        rhs_slice.data().update_slice(rhs.data().slice_scalar_at(unflattenIndex(
                                          rhs_linear_index, rhs.shape())),
                                      output_index);
      }
    });

    // Work on current slice
    auto mul_ret = hal::mul(ctx, lhs_slice, rhs_slice);
    if (ret.dtype() == DT_INVALID) {
      ret = mul_ret;
    } else {
      ret = hal::add(ctx, mul_ret, ret);
    }
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  return ret;
}

spu::Value extractImagePatches(SPUContext *ctx, spu::Value &input,
                               int64_t kernel_x, int64_t kernel_y,
                               int64_t stride_x, int64_t stride_y) {
  auto input_batch = input.shape()[0];
  auto input_x = input.shape()[1];
  auto input_y = input.shape()[2];
  auto input_channels = input.shape()[3];

  std::vector<spu::Value> images;

  for (int64_t x = 0; x <= input_x - kernel_x; x += stride_x) {
    for (int64_t y = 0; y <= input_y - kernel_y; y += stride_y) {
      auto slice = hal::slice(
          ctx, input, {0, x, y, 0},
          {input_batch, x + kernel_x, y + kernel_y, input_channels}, {});
      auto reshaped = hal::reshape(
          ctx, slice, {input_batch, 1, kernel_x, kernel_y, input_channels});
      images.emplace_back(std::move(reshaped));
    }
  }

  auto stacked = hal::concatenate(ctx, images, 1);
  return hal::reshape(
      ctx, stacked,
      {input_batch, stacked.shape()[1], kernel_x * kernel_y, input_channels});
}

// This is an optimized conv2D with im2col
spu::Value Convolution2D(SPUContext *ctx, spu::Value input,
                         const spu::Value &kernel,
                         const ConvolutionConfig &config,
                         absl::Span<const int64_t> result_shape) {
  auto input_batch = input.shape()[0];

  auto kernel_x = kernel.shape()[0];
  auto kernel_y = kernel.shape()[1];
  auto kernel_channels = kernel.shape()[2];
  auto kernel_filters = kernel.shape()[3];

  auto output_x = result_shape[1];
  auto output_y = result_shape[2];

  if (ctx->config().protocol() == ProtocolKind::CHEETAH && input.isSecret() &&
      kernel.isSecret()) {
    // NOTE(juhou): ad-hoc optimization for the current 2PC conv2d
    // implementation. When input_batch is large or small kernel size, it would
    // be better to compute im2col because the current conv2d implementation
    // needs `input_batch` iterations to handle batched input.
    if (input_batch <= kernel_x * kernel_y) {
      return hal::conv2d(ctx, input, kernel, config.window_strides,
                         result_shape);
    }
  }

  std::vector<int64_t> pre_contract_dims{output_y * output_x * input_batch,
                                         kernel_channels * kernel_y * kernel_x};

  std::vector<int64_t> kernel_dims{kernel_channels * kernel_y * kernel_x,
                                   kernel_filters};

  spu::Value extracted_patches =
      extractImagePatches(ctx, input, kernel_x, kernel_y,
                          config.window_strides[0], config.window_strides[1]);

  auto reshaped_patches =
      hal::reshape(ctx, extracted_patches, pre_contract_dims);
  auto reshaped_kernel = hal::reshape(ctx, kernel, kernel_dims);

  auto ret = hal::matmul(ctx, reshaped_patches, reshaped_kernel);

  return hal::reshape(ctx, ret, result_shape);
}

}  // namespace spu::kernel::hlo
