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

namespace spu::mpc::cheetah {
class RandA : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override;
};

class P2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2a"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2p"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "a2v"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override;
};

class V2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2a"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class NegateA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

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

class MulAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};
/// Kernels that idenetical to semi2k ///

class MulA1B : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_a1b"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& ashr,
                  const NdArrayRef& bshr) const override;
};

class MulA1BV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_a1bv"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& ashr,
                  const NdArrayRef& bshr) const override;
};

class MulAA : public BinaryKernel {
 private:
  NdArrayRef mulDirectly(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const;

  NdArrayRef mulWithBeaver(KernelEvalContext* ctx, const NdArrayRef& lhs,
                           const NdArrayRef& rhs) const;

  NdArrayRef squareDirectly(KernelEvalContext* ctx, const NdArrayRef& x) const;

 public:
  static constexpr const char* kBindName() { return "mul_aa"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class SquareA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "square_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

class MulAV : public BinaryKernel {
 private:
  NdArrayRef mulDirectly(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const;

 public:
  static constexpr const char* kBindName() { return "mul_av"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
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

class MatMulAV : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_av"; }

  Kind kind() const override { return Kind::Dynamic; }
  // LHS: m x k
  // RHS: k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatMulVVS : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_vvs"; }

  Kind kind() const override { return Kind::Dynamic; }
  // LHS: m x k
  // RHS: k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_aa"; }

  Kind kind() const override { return Kind::Dynamic; }
  // LHS: m x k
  // RHS: k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class BatchMatMulAV : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "batch_mmul_av";

  Kind kind() const override { return Kind::Dynamic; }

  // override the shape checking
  void evaluate(KernelEvalContext* ctx) const override;
  // LHS: b x m x k
  // RHS: b x k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class Conv2DAA : public Conv2DKernel {
 public:
  static constexpr const char* kBindName() { return "conv2d_aa"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& tensor,
                  const NdArrayRef& filter, int64_t stride_h,
                  int64_t stride_w) const override;
};

class TruncA : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

class LShiftA : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override;
};

class MsbA2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_a2b"; }

  MsbA2B(size_t nbits = 0) : nbits_(nbits) {}

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;

 private:
  size_t nbits_;
};

class EqualAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_aa"; }

  EqualAA(size_t nbits = 0) : nbits_(nbits) {}

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;

 private:
  size_t nbits_;
};

class EqualAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_ap"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class LessAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "f_less_ap"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class LessPA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "f_less_pa"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

}  // namespace spu::mpc::cheetah
