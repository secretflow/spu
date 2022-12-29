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

#include "spu/mpc/object.h"

namespace spu::mpc {

class UnaryKernel : public Kernel {
 public:
  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0)));
  }
  virtual ArrayRef proc(EvalContext* ctx, const ArrayRef& in) const = 0;
};

class ShiftKernel : public Kernel {
 public:
  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<ArrayRef>(0), ctx->getParam<size_t>(1)));
  }
  virtual ArrayRef proc(EvalContext* ctx, const ArrayRef& in,
                        size_t bits) const = 0;
};

class BinaryKernel : public Kernel {
 public:
  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<ArrayRef>(0), ctx->getParam<ArrayRef>(1)));
  }
  virtual ArrayRef proc(EvalContext* ctx, const ArrayRef& lhs,
                        const ArrayRef& rhs) const = 0;
};

class MatmulKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<ArrayRef>(1), ctx->getParam<size_t>(2),
                        ctx->getParam<size_t>(3), ctx->getParam<size_t>(4)));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A,
                        const ArrayRef& B, size_t M, size_t N,
                        size_t K) const = 0;
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

class TruncPrAKernel : public ShiftKernel {
 public:
  virtual bool isPrecise() const = 0;
};

}  // namespace spu::mpc
