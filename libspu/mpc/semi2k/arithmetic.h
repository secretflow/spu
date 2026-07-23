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

namespace spu::mpc::semi2k {

class RandA : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

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

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * (ce::N() - 1); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "a2v"; }

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override;
};

class V2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2a"; }

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class NegateA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_a"; }

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

  ce::CExpr latency() const override {
    // TODO: consider beaver
    return ce::Const(1);
  }

  ce::CExpr comm() const override { return ce::K() * 2 * (ce::N() - 1); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class SquareA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "square_a"; }

  ce::CExpr latency() const override {
    // TODO: consider beaver
    return ce::Const(1);
  }

  ce::CExpr comm() const override { return ce::K() * (ce::N() - 1); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
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
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    auto m = ce::Variable("m", "rows of lhs");
    auto n = ce::Variable("n", "cols of rhs");
    auto k = ce::Variable("k", "cols of lhs");
    return ce::K() * (ce::N() - 1) * (m + n) * k;
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

class TruncA : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }

  // TODO: handle case > 3PC
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return true; }

  // TODO: Add probabilistic truncation (with edabits)
  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Random;
  }
};

// Refer to:
// 5.1 Probabilistic truncation over Z2K, P30,
// Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits
// https://eprint.iacr.org/2020/338.pdf
class TruncAPr : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }

  Kind kind() const override { return Kind::Static; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * (ce::N() - 1); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

class BeaverCacheKernel : public Kernel {
 public:
  static constexpr const char* kBindName() { return "beaver_cache"; }

  void evaluate(KernelEvalContext* ctx) const override;
};

}  // namespace spu::mpc::semi2k
