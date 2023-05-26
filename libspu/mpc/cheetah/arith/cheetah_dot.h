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

#include <memory>

#include "yacl/link/context.h"

#include "libspu/core/array_ref.h"
#include "libspu/mpc/cheetah/arith/common.h"

namespace spu::mpc::cheetah {

// Implementation for Dot.
// Ref: Huang et al. "Cheetah: Lean and Fast Secure Two-Party Deep Neural
// Network Inference"
//  https://eprint.iacr.org/2022/207.pdf
class CheetahDot {
 public:
  explicit CheetahDot(std::shared_ptr<yacl::link::Context> lctx);

  ~CheetahDot();

  CheetahDot& operator=(const CheetahDot&) = delete;

  CheetahDot(const CheetahDot&) = delete;

  CheetahDot(CheetahDot&&) = delete;

  ArrayRef DotOLE(const ArrayRef& inp, yacl::link::Context* conn,
                  const Shape3D& dim3, bool is_left_hand_side);

  ArrayRef DotOLE(const ArrayRef& inp, const Shape3D& dim3,
                  bool is_left_hand_side);

  ArrayRef Conv2dOLE(const ArrayRef& inp, yacl::link::Context* conn,
                     int64_t num_input, const Shape3D& tensor_shape,
                     int64_t num_kernels, const Shape3D& kernel_shape,
                     const Shape2D& window_strides, bool is_tensor);

  ArrayRef Conv2dOLE(const ArrayRef& inp, int64_t num_input,
                     const Shape3D& tensor_shape, int64_t num_kernels,
                     const Shape3D& kernel_shape, const Shape2D& window_strides,
                     bool is_tensor);

  std::shared_ptr<yacl::link::Context> GetLink() const { return lctx_; }

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_{nullptr};

  std::shared_ptr<yacl::link::Context> lctx_{nullptr};
};

}  // namespace spu::mpc::cheetah
