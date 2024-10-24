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

#include "libspu/core/value.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

// This is an optimized conv2D with im2col
spu::Value Convolution2D(SPUContext *ctx, const spu::Value &input,
                         const spu::Value &kernel,
                         const ConvolutionConfig &config,
                         const Shape &result_shape) {
  SPU_ENFORCE(!input.isComplex() && !kernel.isComplex());
  // input  : (N, H, W, C)
  // kernel : (h, w, C, O)
  // output : (N, hh,ww,O), where hh=(H-h)/sh+1, ww=(W-w)/sw+1

  // Alias input dimensions.
  auto N = input.shape()[0];
  auto H = input.shape()[1];
  auto W = input.shape()[2];
  auto C = input.shape()[3];

  auto h = kernel.shape()[0];
  auto w = kernel.shape()[1];
  SPU_ENFORCE_EQ(kernel.shape()[2], C, "input/kernel channel mismatch");
  auto O = kernel.shape()[3];

  SPU_ENFORCE_EQ(result_shape[0], N, "result batch mismatch");
  auto hh = result_shape[1];
  auto ww = result_shape[2];
  SPU_ENFORCE_EQ(result_shape[3], O, "result filters mismatch");

  SPU_ENFORCE_EQ(config.window_strides.size(), 2U);
  int64_t sh = config.window_strides[0];
  int64_t sw = config.window_strides[1];

  SPU_ENFORCE_EQ(hh, (H - h) / sh + 1);
  SPU_ENFORCE_EQ(ww, (W - w) / sw + 1);

  // Fallback, use im2col + dot to implement convolution
  {
    // expand the image according to the kernel size.
    // assumption:
    // - padding is erased by some compiler pass.
    // - input  : NxHxWxC
    // - kernel : hxwxCxO
    Value expanded;
    {
      std::vector<spu::Value> images;
      for (int64_t x = 0; x <= H - h; x += sh) {
        for (int64_t y = 0; y <= W - w; y += sw) {
          auto window =
              hal::slice(ctx, input, {0, x, y, 0}, {N, x + h, y + w, C}, {});
          images.emplace_back(hal::reshape(ctx, window, {N, 1, h, w, C}));
        }
      }
      auto stacked = hal::concatenate(ctx, images, 1);
      SPU_ENFORCE_EQ(stacked.shape()[1], hh * ww);
      expanded = hal::reshape(ctx, stacked, {N, hh * ww, h * w, C});
    }

    // TODO(jint): the below method is much slower then the code above, consider
    // to use slice+reshape+concat to rewrite expandWindow.
    //
    // std::vector<std::pair<int64_t, int64_t>> padding(4, {0, 0});
    // auto expanded = expandWindow(ctx, input,      // input
    //                                    {N, h, w, C},    // window_shape
    //                                    {1, sh, sw, 1},  // strides
    //                                    padding);

    // Now expanded shape is (N, hh*ww, h*w, C)
    SPU_ENFORCE_EQ(expanded.shape()[0], N);
    SPU_ENFORCE_EQ(expanded.shape()[1], hh * ww);
    SPU_ENFORCE_EQ(expanded.shape()[2], h * w);
    SPU_ENFORCE_EQ(expanded.shape()[3], C);

    // Reshape it to (N, hh, ww, h, w, C)
    expanded = hal::reshape(ctx, expanded, {N, hh, ww, h, w, C});

    // Contract on h, w, C
    // expanded:  (N, hh, ww, h, w, C)
    // kernel:               (h, w, C, O)
    // result:    (N, hh, ww,          O)
    auto result = hal::tensordot(ctx, expanded, kernel, {3, 4, 5}, {0, 1, 2});
    SPU_ENFORCE_EQ(result.shape()[0], N);
    SPU_ENFORCE_EQ(result.shape()[1], hh);
    SPU_ENFORCE_EQ(result.shape()[2], ww);
    SPU_ENFORCE_EQ(result.shape()[3], O);

    return result;
  }
}

}  // namespace spu::kernel::hlo
