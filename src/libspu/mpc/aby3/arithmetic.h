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

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/kernel.h"

// Only turn mask on in debug build
#ifndef NDEBUG
#define ENABLE_MASK_DURING_ABY3_P2A
#endif

namespace spu::mpc::aby3 {

class A2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2p"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class P2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2a"; }

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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "a2v"; }

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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override;
};

class V2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2a"; }

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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class RandA : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override;
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
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class MulA1B : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_a1b"; }

  ce::CExpr latency() const override { return ce::Const(2); }

  ce::CExpr comm() const override { return 8 * ce::K(); }

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
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    auto m = ce::Variable("m", "rows of lhs");
    auto n = ce::Variable("n", "cols of rhs");
    return ce::K() * m * n;
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

// Refer to:
// Share Truncation I, 5.1 Fixed-point Arithmetic, P13,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
class TruncA : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
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
  static constexpr const char* kBindName() { return "trunc_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(2); }

  // the comm. is indeed asymmetric.
  // let k be the field size, bw be the bitwidth of truncation.
  // Of course, k, bw will be one of 32,64,128 now.
  // 1. if sign is unknown:  send (4k,2k,2k)     with (2,2,2) rounds.
  // 2. if sign is positive: send (k+bw,k+bw,bw) with (2,2,1) rounds.
  // 3. if sign is negative: send (k+bw,k+bw,bw) with (2,2,1) rounds.
  ce::CExpr comm() const override { return 2 * ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

// Ref: Improved secure two-party computation from a geometric perspective
// https://eprint.iacr.org/2025/200
// Algorithm 4: One-bit error truncation with constraint
// NOTE: this algorithm needs |x| < L / 4, where L = 2^l, and l is the
// bit-length of the field. Fortunately, this condition is always satisfied
// under current SPU encoding scheme, see /libspu/core/encoding.cc for more
// details.
class TruncAPr2 : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }

  static constexpr size_t kBitsLeftOut = 2;

  // the communication is dependent on the truncation bits.
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(2); }

  // the comm. is indeed asymmetric.
  // let k be the field size, bw be the bitwidth of truncation.
  // Of course, k, bw will be one of 32,64,128 now.
  // each Party will send (k+bw,k+bw,bw) with (2,2,1) rounds.
  ce::CExpr comm() const override { return 2 * ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

}  // namespace spu::mpc::aby3
