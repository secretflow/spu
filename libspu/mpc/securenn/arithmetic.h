// Copyright 2023 Ant Group Co., Ltd.
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

namespace spu::mpc::securenn {

class A2V : public RevealToKernel {
 public:
  static constexpr char kBindName[] = "a2v";

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override;
};

class V2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "v2a";

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class RandA : public RandKernel {
 public:
  static constexpr char kBindName[] = "rand_a";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override;
};

class P2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2a";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2p";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * (ce::N() - 1); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class NotA : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_a";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
class AddAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ap";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class AddAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_aa";

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
  static constexpr char kBindName[] = "mul_ap";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class MulAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_aa";

  ce::CExpr latency() const override {
    // online
    return ce::Const(1);
  }

  ce::CExpr comm() const override { return ce::K() * 4; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
class MatMulAP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ap";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class LShiftA : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_a";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t bits) const override;
};

// Refer to:
// 5.1 Probabilistic truncation over Z2K, P30,
// Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits
// https://eprint.iacr.org/2020/338.pdf
class TruncAPr : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_a";

  Kind kind() const override { return Kind::Static; }
  // offline + online
  ce::CExpr latency() const override { return ce::Const(5); }

  ce::CExpr comm() const override { return ce::K() * ce::Const(5); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

class MatMulAA : public MatmulKernel {
 public:
  // static constexpr char kBindName[] = "mmul_aa_2pc";
  static constexpr char kBindName[] = "mmul_aa";

  ce::CExpr latency() const override {
    // beaver + online
    return ce::Const(2);
  }

  ce::CExpr comm() const override {
    // beaver + online
    auto m = ce::Variable("m", "rows of lhs");
    auto n = ce::Variable("n", "cols of rhs");
    auto k = ce::Variable("k", "cols of lhs");
    return ce::K() * (2 * m * k + 2 * k * n);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class MatMulAA_simple : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_aa_simple";

  ce::CExpr latency() const override {
    // beaver + online
    return ce::Const(2);
  }

  ce::CExpr comm() const override {
    // beaver + online
    auto m = ce::Variable("m", "rows of lhs");
    auto n = ce::Variable("n", "cols of rhs");
    auto k = ce::Variable("k", "cols of lhs");
    return ce::K() * (2 * m * k + 2 * k * n);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class Msb : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2a";

  ce::CExpr latency() const override { return ce::Const(5); }
  ce::CExpr comm() const override {
    const auto log_p =
        9;  // in fact, now the element is ring2k_t rather than [0, p-1]
    return (13 * ce::K() + 4 * ce::K() * log_p);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class Msb_opt : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_opt_a2a";

  ce::CExpr latency() const override { return ce::Const(5); }
  ce::CExpr comm() const override {
    const auto log_p =
        9;  // in fact, now the element is ring2k_t rather than [0, p-1]
    return (9 * ce::K() + 3 * ce::K() * log_p);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class ShareConvert : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "sc";
  ce::CExpr latency() const override { return ce::Const(4); }
  ce::CExpr comm() const override {
    const auto log_p = 9;
    return (6 * ce::K() + 4 * log_p * ce::K());
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

}  // namespace spu::mpc::securenn
