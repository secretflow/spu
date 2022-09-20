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

#include "spu/device/pphlo/kernels/convolution.h"

#include "spu/device/pphlo/kernels/utils.h"
#include "spu/hal/hal.h"

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

} // namespace

namespace spu::device::pphlo::kernel {

hal::Value Convolution(HalContext *ctx, const hal::Value &lhs,
                       const hal::Value &rhs, const ConvolutionConfig &config,
                       absl::Span<const int64_t> result_shape) {
  const size_t num_spatial_dims = config.outputSpatialDimensions.size();
  YASL_ENFORCE(num_spatial_dims == config.inputSpatialDimensions.size());
  YASL_ENFORCE(num_spatial_dims == config.kernelSpatialDimensions.size());

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
  hal::Value ret;

  // Iterate on window
  std::vector<int64_t> window_index(config.kernelSpatialDimensions.size(), 0);

  do {
    hal::Value lhs_slice =
        hal::zeros(ctx, lhs.vtype(), lhs.dtype(), result_shape);
    hal::Value rhs_slice =
        hal::zeros(ctx, rhs.vtype(), rhs.dtype(), result_shape);

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
        const int64_t undilated_index =
            output_index[output_spatial_dim] * config.window_strides[ki] -
            config.padding[2 * ki] + window_index[ki] * config.rhs_dilation[ki];
        // Skip if the lhs (input) index is to be dilated.  As an
        // optimization, skip this mod if there's no dilation.
        if (config.lhs_dilation[ki] > 1 &&
            undilated_index % config.lhs_dilation[ki] != 0) {
          return;
        }

        // Calculate the actual lhs (input) index after dilation.  As an
        // optimization, skip this integer divide if there's no dilation.
        int64_t lhs_spatial_index;
        if (config.lhs_dilation[ki] > 1) {
          lhs_spatial_index = undilated_index / config.lhs_dilation[ki];
        } else {
          lhs_spatial_index = undilated_index;
        }

        // Skip if input index is not in bounds.
        if (!(lhs_spatial_index >= 0 &&
              lhs_spatial_index < lhs_shape[input_spatial_dim])) {
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

        lhs_slice.copyElementFrom(lhs.getElementAt(lhs_linear_index), {},
                                  output_index);
        rhs_slice.copyElementFrom(rhs.getElementAt(rhs_linear_index), {},
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

// This is an optimized conv2D with im2col
hal::Value Convolution2D(HalContext *ctx, hal::Value input, hal::Value kernel,
                         const ConvolutionConfig &config) {
  // 01io -> o01i
  const auto batch = input.shape()[config.inputBatchDimension];
  const auto feature = input.shape()[config.inputFeatureDimension];
  auto input_h = input.shape()[config.inputSpatialDimensions[0]];
  auto input_w = input.shape()[config.inputSpatialDimensions[1]];
  const auto out = kernel.shape()[config.kernelOutputFeatureDimension];
  auto kernel_h = kernel.shape()[config.kernelSpatialDimensions[0]];
  auto kernel_w = kernel.shape()[config.kernelSpatialDimensions[1]];

  // transpose to b01f_01io
  input = hal::transpose(
      ctx, input,
      {config.inputBatchDimension, config.inputSpatialDimensions[0],
       config.inputSpatialDimensions[1], config.inputFeatureDimension});
  kernel = hal::transpose(ctx, kernel,
                          {config.kernelSpatialDimensions[0],
                           config.kernelSpatialDimensions[1],
                           config.kernelOutputFeatureDimension,
                           config.kernelInputFeatureDimension});

  bool lhs_need_padding = false;

  lhs_need_padding |= std::any_of(config.padding.begin(), config.padding.end(),
                                  [](int64_t i) { return i != 0; });

  lhs_need_padding |=
      std::any_of(config.lhs_dilation.begin(), config.lhs_dilation.end(),
                  [](int64_t i) { return i != 1; });

  if (lhs_need_padding) {
    // add padding
    auto padding_value = hal::zeros(ctx, input.vtype(), input.dtype());
    input = hal::pad(
        ctx, input, padding_value, {0, config.padding[0], config.padding[2], 0},
        {0, config.padding[1], config.padding[3], 0},
        {0, config.lhs_dilation[0] - 1, config.lhs_dilation[1] - 1, 0});
    input_h = input_h + (input_h - 1) * (config.lhs_dilation[0] - 1);
    input_w = input_w + (input_w - 1) * (config.lhs_dilation[1] - 1);
  }

  bool need_dilate =
      std::any_of(config.rhs_dilation.begin(), config.rhs_dilation.end(),
                  [](int64_t i) { return i != 1; });
  if (need_dilate) {
    auto padding_value = hal::zeros(ctx, kernel.vtype(), kernel.dtype());
    kernel = hal::pad(
        ctx, kernel, padding_value, {0, 0, 0, 0}, {0, 0, 0, 0},
        {config.rhs_dilation[0] - 1, config.rhs_dilation[1] - 1, 0, 0});
    kernel_h = kernel_h + (kernel_h - 1) * (config.rhs_dilation[0] - 1);
    kernel_w = kernel_w + (kernel_w - 1) * (config.rhs_dilation[1] - 1);
  }

  YASL_ENFORCE((input_h + config.padding[0] + config.padding[1]) >= kernel_h);
  YASL_ENFORCE((input_w + config.padding[2] + config.padding[3]) >= kernel_w);

  const auto out_h =
      (input_h - kernel_h + config.padding[0] + config.padding[1]) /
          config.window_strides[0] +
      1;
  const auto out_w =
      (input_w - kernel_w + config.padding[2] + config.padding[3]) /
          config.window_strides[1] +
      1;

  std::vector<hal::Value> im2col_elements;

  kernel = hal::reshape(ctx, kernel, {out, feature * kernel_h * kernel_w});
  for (int64_t i = 0;
       i <= input_h - kernel_h + config.padding[0] + config.padding[1];
       i += config.window_strides[0]) {
    for (int64_t j = 0;
         j <= input_w - kernel_w + config.padding[2] + config.padding[3];
         j += config.window_strides[1]) {
      const auto sliced_input = hal::reshape(
          ctx,
          hal::slice(ctx, input, {0, i, j, 0},
                     {batch, i + kernel_h, j + kernel_w, feature}, {}),
          {batch, feature * kernel_h * kernel_w});

      im2col_elements.emplace_back(sliced_input);
    }
  }

  auto im2col = hal::concatenate(ctx, im2col_elements, 1);

  im2col = hal::reshape(ctx, im2col,
                        {batch * out_h * out_w, feature * kernel_h * kernel_w});

  auto ret = hal::matmul(ctx, im2col, transpose(ctx, kernel));

  ret = hal::reshape(ctx, ret, {batch, out_h, out_w, out});

  // Transpose output from b01f
  std::vector<int64_t> permutation(4);
  permutation[config.outputBatchDimension] = 0;
  permutation[config.outputSpatialDimensions[0]] = 1;
  permutation[config.outputSpatialDimensions[1]] = 2;
  permutation[config.outputFeatureDimension] = 3;
  return hal::transpose(ctx, ret, permutation);
}

} // namespace spu::device::pphlo::kernel
