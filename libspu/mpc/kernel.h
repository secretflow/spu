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

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"

namespace spu::mpc {

class RandKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& shape = ctx->getParam<Shape>(0);

    ArrayRef res = proc(ctx, shape.numel());

    ctx->setOutput(WrapValue(res, shape));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, size_t size) const = 0;
};

class UnaryKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<Value>(0);
    auto [arr, shape, dtype] = UnwrapValue(in);

    ArrayRef res = proc(ctx, arr);

    ctx->setOutput(WrapValue(res, shape));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const = 0;
};

class ShiftKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<Value>(0);
    size_t bits = ctx->getParam<size_t>(1);
    auto [arr, shape, dtype] = UnwrapValue(in);

    ArrayRef res = proc(ctx, arr, bits);

    ctx->setOutput(WrapValue(res, shape));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const = 0;
};

class BinaryKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& lhs = ctx->getParam<Value>(0);
    const auto& rhs = ctx->getParam<Value>(1);

    SPU_ENFORCE(lhs.shape() == rhs.shape(), "shape mismatch {} {}", lhs.shape(),
                rhs.shape());

    auto [x, shape, dtype] = UnwrapValue(lhs);
    auto [y, _, _1] = UnwrapValue(rhs);

    auto z = proc(ctx, x, y);

    ctx->setOutput(WrapValue(z, shape));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                        const ArrayRef& rhs) const = 0;
};

class MatmulKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& lhs = ctx->getParam<Value>(0);
    const auto& rhs = ctx->getParam<Value>(1);

    // TODO: drop (m, n, k)
    auto m = static_cast<int64_t>(ctx->getParam<size_t>(2));
    auto n = static_cast<int64_t>(ctx->getParam<size_t>(3));
    auto k = static_cast<int64_t>(ctx->getParam<size_t>(4));

    // SPU_ENFORCE(lhs.shape().size() == 2 && rhs.shape().size() == 2 &&
    //                 lhs.shape()[0] == m && lhs.shape()[1] == k &&
    //                 rhs.shape()[0] == k && rhs.shape()[1] == n,
    //             "invalid shape {} {}", lhs.shape(), rhs.shape());
    SPU_ENFORCE(
        calcNumel(lhs.shape()) == m * k && calcNumel(rhs.shape()) == k * n,
        "invalid shape {} {}", lhs.shape(), rhs.shape());

    auto [x, shape, dtype] = UnwrapValue(lhs);
    auto [y, _, _1] = UnwrapValue(rhs);

    auto z = proc(ctx, x, y, m, n, k);

    ctx->setOutput(WrapValue(z, Shape{m, n}));
  }

  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& a,
                        const ArrayRef& b, size_t m, size_t n,
                        size_t k) const = 0;
};

class Conv2DKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& lhs = ctx->getParam<Value>(0);
    const auto& rhs = ctx->getParam<Value>(1);
    size_t N = ctx->getParam<size_t>(2);
    size_t H = ctx->getParam<size_t>(3);
    size_t W = ctx->getParam<size_t>(4);
    size_t C = ctx->getParam<size_t>(5);
    size_t O = ctx->getParam<size_t>(6);
    size_t h = ctx->getParam<size_t>(7);
    size_t w = ctx->getParam<size_t>(8);
    size_t stride_h = ctx->getParam<size_t>(9);
    size_t stride_w = ctx->getParam<size_t>(10);

    auto [x, shape, dtype] = UnwrapValue(lhs);
    auto [y, _, _1] = UnwrapValue(rhs);

    auto z = proc(ctx, x, y, N, H, W, C, O, h, w, stride_h, stride_w);

    ctx->setOutput(WrapValue(z, shape));
  }
  // tensor: NxHxWxC
  // filter: hxwxCxO
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& tensor,
                        const ArrayRef& filter, size_t N, size_t H, size_t W,
                        size_t C, size_t O, size_t h, size_t w, size_t stride_h,
                        size_t stride_w) const = 0;
};

class BitrevKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<Value>(0);
    size_t start = ctx->getParam<size_t>(1);
    size_t end = ctx->getParam<size_t>(2);

    auto [x, shape, dtype] = UnwrapValue(in);

    auto z = proc(ctx, x, start, end);

    ctx->setOutput(WrapValue(z, shape));
  }

  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t start, size_t end) const = 0;
};

enum class TruncLsbRounding {
  // For protocols like SecureML/ABY3, the LSB is random.
  Random,
  // For DEK19/EGK+20, the LSB is probabilistic, More precisely, for
  //    y = x/2` + u, where u ∈ [0, 1).
  // The result has probability of u to be x/2`+1, and probability 1-u to be
  // x/2`.
  Probabilistic,
  // For some deterministic truncation, the LSB is deterministic.
  Nearest,
};

class TruncAKernel : public ShiftKernel {
 public:
  // For protocol like SecureML, the most significant bit may have error with
  // low probability, which lead to huge calculation error.
  //
  // Return true if the protocol has this kind of error.
  virtual bool hasMsbError() const = 0;

  virtual TruncLsbRounding lsbRounding() const = 0;
};

class TruncAWithSignKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<Value>(0);
    size_t bits = ctx->getParam<size_t>(1);
    bool positive = ctx->getParam<bool>(2);

    auto [x, shape, dtype] = UnwrapValue(in);
    auto z = proc(ctx, x, bits, positive);

    ctx->setOutput(WrapValue(z, shape));
  }

  virtual bool hasMsbError() const = 0;

  virtual TruncLsbRounding lsbRounding() const = 0;

  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t bits,
                        bool is_positive) const = 0;
};

class BitSplitKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<Value>(0);
    size_t stride = ctx->getParam<size_t>(1);
    auto [arr, shape, dtype] = UnwrapValue(in);

    ArrayRef res = proc(ctx, arr, stride);

    ctx->setOutput(WrapValue(res, shape));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t stride) const = 0;
};

class CastTypeKernel : public Kernel {
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& val = ctx->getParam<Value>(0);
    const auto& to_type = ctx->getParam<Type>(1);

    auto [arr, shape, dtype] = UnwrapValue(val);

    ArrayRef res = proc(ctx, arr, to_type);

    ctx->setOutput(WrapValue(res, shape));
  }

  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                        const Type& to_type) const = 0;
};

}  // namespace spu::mpc
