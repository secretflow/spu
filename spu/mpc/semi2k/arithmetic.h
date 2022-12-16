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

class ZeroA : public Kernel {
 public:
  static constexpr char kBindName[] = "zero_a";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<FieldType>(0), ctx->getParam<size_t>(1)));
  }

  ArrayRef proc(KernelEvalContext* ctx, FieldType field, size_t size) const;
};

class RandA : public Kernel {
 public:
  static constexpr char kBindName[] = "rand_a";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<size_t>(0)));
  }

  ArrayRef proc(KernelEvalContext* ctx, size_t size) const;
};

class P2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2a";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class A2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2p";

  CExpr latency() const override { return Const(1); }

  CExpr comm() const override { return K() * (N() - 1); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class NotA : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_a";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
class AddAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ap";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AddAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_aa";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
class MulAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_ap";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MulAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_aa";

  CExpr latency() const override {
    // TODO: consider beaver
    return Const(1);
  }

  CExpr comm() const override { return K() * 2 * (N() - 1); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
class MatMulAP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ap";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                size_t M, size_t N, size_t K) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_aa";

  // TODO(jint) express M, N, K
  Kind kind() const override { return Kind::kDynamic; }

  CExpr latency() const override { return nullptr; }

  CExpr comm() const override { return nullptr; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                size_t M, size_t N, size_t K) const override;
};

class LShiftA : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_a";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class TruncPrA : public TruncPrAKernel {
 public:
  static constexpr char kBindName[] = "truncpr_a";

  // TODO: handle case > 3PC
  Kind kind() const override { return Kind::kDynamic; }

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;

  bool isPrecise() const override { return false; }
};

}  // namespace spu::mpc::semi2k
