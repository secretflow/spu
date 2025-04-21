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

#ifndef NONLINE
#define ONLINE_ONLY
#endif

namespace spu::mpc::shamir {

class A2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2b"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(10); }

  ce::CExpr comm() const override {
    auto nBits = ce::Variable("nBits", "number of Bits");
    auto sum = ce::Variable("sum", "sum_{i=1}^{nBits} (4i-2)");
    return ce::K() * (2 + 6 * nBits + 2 * sum);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

class B2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

// Todo: put this kernel to arithmetic.h
class TruncA : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(2); }

  // only count online for now.
  ce::CExpr comm() const override { return ce::K() * 2; }

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
  static constexpr const char* kBindName() { return "mul_aa_trunc"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(2); }

  // only count online for now.
  ce::CExpr comm() const override { return ce::K() * 2; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  // TODO: Add probabilistic truncation (with edabits)
  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Random;
  }
};

class MsbA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_a"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(6); }

  // only count online for now.
  ce::CExpr comm() const override {
    auto nBits = ce::Variable("nBits", "number of Bits");
    return 2 * ce::K() + 2 * ce::K() * nBits;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class ReLU : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "relu"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(6); }

  // only count online for now.
  ce::CExpr comm() const override {
    auto nBits = ce::Variable("nBits", "number of Bits");
    return 4 * ce::K() + 2 * ce::K() * nBits;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class CommonTypeV : public Kernel {
 public:
  static constexpr const char* kBindName() { return "common_type_v"; }

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override;
};

}  // namespace spu::mpc::shamir
