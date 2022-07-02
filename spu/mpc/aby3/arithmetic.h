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

#include "spu/core/array_ref.h"
#include "spu/mpc/kernel.h"

namespace spu::mpc::aby3 {

using util::CExpr;
using util::Const;
using util::K;
using util::Log;
using util::N;

#define ENABLE_MASK_DURING_ABY3_P2A

class A2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2p";

  CExpr latency() const override {
    // 1 * rotate: 1
    return Const(1);
  }

  CExpr comm() const override {
    // 1 * rotate: k
    return K();
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class P2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2a";

  CExpr latency() const override {
#ifdef ENABLE_MASK_DURING_ABY3_P2A
    return Const(1);
#else
    return Const(0);
#endif
  }

  CExpr comm() const override {
#ifdef ENABLE_MASK_DURING_ABY3_P2A
    return K();
#else
    return Const(0);
#endif
  }

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
    // 1 * rotate: 1
    return Const(1);
  }

  CExpr comm() const override {
    // 1 * rotate: k
    return K();
  }

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

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x, const ArrayRef& y,
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

// Refer to:
// Share Truncation I, 5.1 Fixed-point Arithmetic, P13,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
class TruncPrA : public TruncPrAKernel {
 public:
  static constexpr char kBindName[] = "truncpr_a";

  CExpr latency() const override { return Const(1); }

  CExpr comm() const override { return K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;

  bool isPrecise() const override { return false; }
};

// Refer to:
// 3.2.2 Truncation by a public value, P10,
// Secure Evaluation of Quantized Neural Networks
// - https://arxiv.org/pdf/1910.12435.pdf
class TruncPrAPrecise : public TruncPrAKernel {
 public:
  static constexpr char kBindName[] = "truncpr_a";

  CExpr latency() const override { return Const(3); }

  CExpr comm() const override { return 4 * K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;

  bool isPrecise() const override { return true; }
};

}  // namespace spu::mpc::aby3
