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

#include "libspu/mpc/kernel.h"

namespace spu::mpc::standard_shape {

class Broadcast : public BroadcastKernel {
 public:
  static constexpr const char* kBindName() { return "broadcast"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Shape& to_shape, const Axes& in_dims) const override;
};

class Reshape : public ShapeBasedKernel {
 public:
  static constexpr const char* kBindName() { return "reshape"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Shape& to_shape) const override;
};

class ExtractSlice : public ExtractSliceKernel {
 public:
  static constexpr const char* kBindName() { return "extract_slice"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Index& start, const Index& end,
                  const Strides& strides) const override;
};

class UpdateSlice : public UpdateSliceKernel {
 public:
  static constexpr const char* kBindName() { return "update_slice"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& update, const Index& start) const override;
};

class Transpose : public DimsBasedKernel {
 public:
  static constexpr const char* kBindName() { return "transpose"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Axes& permutation) const override;
};

class Reverse : public DimsBasedKernel {
 public:
  static constexpr const char* kBindName() { return "reverse"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Axes& dimensions) const override;
};

class Fill : public ShapeBasedKernel {
 public:
  static constexpr const char* kBindName() { return "fill"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Shape& to_shape) const override;
};

class Pad : public PadKernel {
 public:
  static constexpr const char* kBindName() { return "pad"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& padding_value,
                  const Sizes& edge_padding_low, const Sizes& edge_padding_high,
                  const Sizes& interior_padding) const override;
};

class Concate : public ConcateKernel {
 public:
  static constexpr const char* kBindName() { return "concatenate"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const std::vector<NdArrayRef>& vales,
                  int64_t axis) const override;
};
}  // namespace spu::mpc::standard_shape
