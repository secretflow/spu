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

namespace spu::mpc::shamir {

class A2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2b";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override {
    return (Log(ce::K()) + 1)  // adder-circuit;
           * Log(ce::N())      // tree-reduce parties.
        ;
  }

  ce::CExpr comm() const override {
    return (2 * Log(ce::K()) + 1)         // KS-adder-circuit
           * 2 * ce::K() * (ce::N() - 1)  // And gate, for nPC
           * (ce::N() - 1)                // (no-matter tree or ring) reduce
        ;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

class B2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override {
    return ce::Const(0);
  }

  ce::CExpr comm() const override {
    return ce::Const(0);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

// Todo: put this kernel to arithmetic.h
class TruncA : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_a";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  // TODO: Add probabilistic truncation (with edabits)
  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Random;
  }
};

class MulAATrunc : public MulTruncAKernel {
 public:
  static constexpr char kBindName[] = "mul_aa_trunc";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, const NdArrayRef& y, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  // TODO: Add probabilistic truncation (with edabits)
  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Random;
  }
};

class MsbA : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override {
    // 1 * carry : log(k) + 1
    // 1 * rotate: 1
    return Log(ce::K()) + 1 + 1;
  }

  ce::CExpr comm() const override {
    // 1 * carry : k + 2 * k + 16 * 2
    // 1 * rotate: k
    return ce::K() + 2 * ce::K() + ce::K() + 32;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class CommonTypeV : public Kernel {
 public:
  static constexpr char kBindName[] = "common_type_v";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override;
};

}  // namespace spu::mpc::shamir
