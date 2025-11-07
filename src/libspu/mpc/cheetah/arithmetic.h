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

class MixMulAA : public MixBitMulKernel {
 private:
  // SIRNN's version
  NdArrayRef mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y, SignType sign_x, SignType sign_y,
                         FieldType to_field, size_t out_bw,
                         bool signed_arith) const;

  // naive extend first then mul
  NdArrayRef mulNaively(KernelEvalContext* ctx, const NdArrayRef& x,
                        const NdArrayRef& y, SignType sign_x, SignType sign_y,
                        FieldType to_field, size_t out_bw,
                        bool signed_arith) const;

 public:
  static constexpr const char* kBindName() { return "mixmul_aa"; }

  Kind kind() const override { return Kind::Dynamic; }

  // 1. if out_bw=0 and to_field=FT_INVALID, raise error
  // 2. if to_field=FT_INVALID, but out_bw>0, then to_field is choosen as the
  // minimum field
  // 3. if out_bw=0, to_field != FT_INVALID, just use to_field
  // 4. else, first check if to_field is valid
  NdArrayRef proc(KernelEvalContext* ctx,
                  const NdArrayRef& x,                         //
                  const NdArrayRef& y,                         //
                  SignType sign_x,                             //
                  SignType sign_y,                             //
                  FieldType to_field = FieldType::FT_INVALID,  //
                  size_t out_bw = 0,                           //
                  bool signed_arith = true                     //
  ) const override;
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

class TruncA2 : public TruncAWithSignedKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a2"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, size_t bits,
                  SignType sign, bool signed_arith = true) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

class TruncAE : public TruncAWithSignedKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_ae"; }

  Kind kind() const override { return Kind::Dynamic; }

  // TODO: I don't know why cheetah trunc_ae has a error of 1 bit...
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, size_t bits,
                  SignType sign, bool signed_arith = true) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Exact;
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

class MsbEq : public CmpEqKernel {
 public:
  static constexpr const char* kBindName() { return "msb_eq"; }

  Kind kind() const override { return Kind::Dynamic; }

  std::vector<NdArrayRef> proc(KernelEvalContext* ctx,
                               const NdArrayRef& x) const override;
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

class LutAP : public LookUpTableKernel {
 public:
  static constexpr const char* kBindName() { return "lut_ap"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& table, size_t bw,
                  FieldType field = FieldType::FT_INVALID) const override;
};

}  // namespace spu::mpc::cheetah
