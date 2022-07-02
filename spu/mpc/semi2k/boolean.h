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

#include "spu/mpc/kernel.h"
#include "spu/mpc/util/cexpr.h"

namespace spu::mpc::semi2k {

using util::CExpr;
using util::Const;
using util::K;
using util::Log;
using util::N;

class ZeroB : public Kernel {
 public:
  static constexpr char kBindName[] = "zero_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<FieldType>(0), ctx->getParam<size_t>(1)));
  }

  ArrayRef proc(KernelEvalContext* ctx, FieldType field, size_t size) const;
};

class B2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2p";

  CExpr latency() const override { return Const(1); }

  CExpr comm() const override { return K() * (N() - 1); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class P2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class AndBP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_bp";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AndBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_bb";

  CExpr latency() const override { return Const(1); }

  CExpr comm() const override { return K() * 2 * (N() - 1); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class XorBP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_bp";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class XorBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_bb";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class LShiftB : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class RShiftB : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ARShiftB : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class BitrevB : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override;
};

}  // namespace spu::mpc::semi2k
