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

#include "libspu/core/memref.h"
#include "libspu/mpc/kernel.h"

// Only turn mask on in debug build
#ifndef NDEBUG
#define ENABLE_MASK_DURING_ABY3_P2A
#endif

namespace spu::mpc::aby3 {

class CommonTypeA : public Kernel {
 public:
  static constexpr const char *kBindName() { return "common_type_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  void evaluate(KernelEvalContext *ctx) const override;
};

class CastTypeA : public CastTypeKernel {
 public:
  static constexpr const char *kBindName() { return "cast_type_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in,
              const Type &to_type) const override;
};

class A2P : public UnaryKernel {
 public:
  static constexpr const char *kBindName() { return "a2p"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in) const override;
};

class P2A : public UnaryKernel {
 public:
  static constexpr const char *kBindName() { return "p2a"; }

  ce::CExpr latency() const override {
#ifdef ENABLE_MASK_DURING_ABY3_P2A
    return ce::Const(1);
#else
    return ce::Const(0);
#endif
  }

  ce::CExpr comm() const override {
#ifdef ENABLE_MASK_DURING_ABY3_P2A
    return ce::K();
#else
    return ce::Const(0);
#endif
  }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in) const override;
};

class A2V : public RevealToKernel {
 public:
  static constexpr const char *kBindName() { return "a2v"; }

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override {
    // 1 * send/recv: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in,
              size_t rank) const override;
};

class V2A : public UnaryKernel {
 public:
  static constexpr const char *kBindName() { return "v2a"; }

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in) const override;
};

class RandA : public RandKernel {
 public:
  static constexpr const char *kBindName() { return "rand_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, SemanticType type,
              const Shape &shape) const override;
};

class NegateA : public UnaryKernel {
 public:
  static constexpr const char *kBindName() { return "negate_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in) const override;
};

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
class AddAP : public BinaryKernel {
 public:
  static constexpr const char *kBindName() { return "add_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &lhs,
              const MemRef &rhs) const override;
};

class AddAA : public BinaryKernel {
 public:
  static constexpr const char *kBindName() { return "add_aa"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &lhs,
              const MemRef &rhs) const override;
};

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
class MulAP : public BinaryKernel {
 public:
  static constexpr const char *kBindName() { return "mul_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &lhs,
              const MemRef &rhs) const override;
};

class MulAA : public BinaryKernel {
 public:
  static constexpr const char *kBindName() { return "mul_aa"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  MemRef proc(KernelEvalContext *ctx, const MemRef &lhs,
              const MemRef &rhs) const override;
};

class MulA1B : public BinaryKernel {
 public:
  static constexpr const char *kBindName() { return "mul_a1b"; }

  ce::CExpr latency() const override { return ce::Const(2); }

  ce::CExpr comm() const override { return 8 * ce::K(); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &lhs,
              const MemRef &rhs) const override;
};

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
class MatMulAP : public MatmulKernel {
 public:
  static constexpr const char *kBindName() { return "mmul_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &x,
              const MemRef &y) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr const char *kBindName() { return "mmul_aa"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    auto m = ce::Variable("m", "rows of lhs");
    auto n = ce::Variable("n", "cols of rhs");
    return ce::K() * m * n;
  }

  MemRef proc(KernelEvalContext *ctx, const MemRef &x,
              const MemRef &y) const override;
};

class LShiftA : public ShiftKernel {
 public:
  static constexpr const char *kBindName() { return "lshift_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in,
              const Sizes &bits) const override;
};

// Refer to:
// Share Truncation I, 5.1 Fixed-point Arithmetic, P13,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
class TruncA : public TruncAKernel {
 public:
  static constexpr const char *kBindName() { return "trunc_a"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in, size_t bits,
              SignType sign) const override;

  bool hasMsbError() const override { return true; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Random;
  }
};

// Refer to:
// 3.2.2 Truncation by a public value, P10,
// Secure Evaluation of Quantized Neural Networks
// - https://arxiv.org/pdf/1910.12435.pdf
class TruncAPr : public TruncAKernel {
 public:
  static constexpr const char *kBindName() { return "trunc_a"; }

  ce::CExpr latency() const override { return ce::Const(3); }

  ce::CExpr comm() const override { return 4 * ce::K(); }

  MemRef proc(KernelEvalContext *ctx, const MemRef &in, size_t bits,
              SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

}  // namespace spu::mpc::aby3
