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

#include "libspu/mpc/object.h"

namespace spu::mpc {

class UnaryKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0)));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const = 0;
};

class ShiftKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<ArrayRef>(0), ctx->getParam<size_t>(1)));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const = 0;
};

class BinaryKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<ArrayRef>(0), ctx->getParam<ArrayRef>(1)));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                        const ArrayRef& rhs) const = 0;
};

class MatmulKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<ArrayRef>(1), ctx->getParam<size_t>(2),
                        ctx->getParam<size_t>(3), ctx->getParam<size_t>(4)));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& a,
                        const ArrayRef& b, size_t m, size_t n,
                        size_t k) const = 0;
};

class Conv2DKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<ArrayRef>(1), ctx->getParam<size_t>(2),
                        ctx->getParam<size_t>(3), ctx->getParam<size_t>(4),
                        ctx->getParam<size_t>(5), ctx->getParam<size_t>(6),
                        ctx->getParam<size_t>(7), ctx->getParam<size_t>(8),
                        ctx->getParam<size_t>(9), ctx->getParam<size_t>(10)));
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
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<size_t>(1), ctx->getParam<size_t>(2)));
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
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<size_t>(1), ctx->getParam<bool>(2)));
  }

  virtual bool hasMsbError() const = 0;

  virtual TruncLsbRounding lsbRounding() const = 0;

  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t bits,
                        bool is_positive) const = 0;
};

}  // namespace spu::mpc
