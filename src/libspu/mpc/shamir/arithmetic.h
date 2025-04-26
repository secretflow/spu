// Copyright 2024 Ant Group Co., Ltd.
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

#ifndef NDEBUG
#define ENABLE_MASK_DURING_SHAMIR_P2A
#endif

#ifndef NONLINE
#define ONLINE_ONLY
#endif

namespace spu::mpc::shamir {

class RandA : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override {
    auto dy_times = ce::Variable(
        "D", "dynamic_times = ceil(numl / (n-t)) = [(numel – 1) / (n-t) + 1]");
    return (ce::N() - 1) * ce::K() * dy_times;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override;
};

class P2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2p"; }

  ce::CExpr latency() const override { return ce::Const(2); }

  ce::CExpr comm() const override { return ce::K() * 2; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "a2v"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override;
};

class V2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2a"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class NegateA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
class AddAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class AddAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_aa"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
class MulAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class MulAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_aa"; }

  ce::CExpr latency() const override { return ce::Const(2); }

  ce::CExpr comm() const override { return ce::K() * 2; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class MulAAA : public TernaryKnernel {
 public:
  static constexpr const char* kBindName() { return "mul_aaa"; }

  ce::CExpr latency() const override { return ce::Const(2); }

  ce::CExpr comm() const override { return ce::K() * 4; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y, const NdArrayRef& z) const override;
};

class MulAAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_aa_p"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * 2; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
class MatMulAP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_aa"; }

  ce::CExpr latency() const override {
    // only count online for now.
    return ce::Const(2);
  }

  ce::CExpr comm() const override {
    auto m = ce::Variable("m", "rows of lhs");
    auto n = ce::Variable("n", "cols of rhs");
    return ce::K() * 2 * (m * n);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class LShiftA : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override;
};

}  // namespace spu::mpc::shamir
