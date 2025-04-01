// Copyright 2025 Ant Group Co., Ltd.
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

namespace spu::mpc::fantastic4 {

class A2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2b"; }

  ce::CExpr latency() const override {
    return Log(ce::K()) + 1 + 1;
  }

  // TODO: this depends on the adder circuit.
  ce::CExpr comm() const override {
    return 2 * Log(ce::K()) * ce::K() + ce::K() * 2;
  }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class B2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }

  ce::CExpr latency() const override {
    return ce::Const(3) + Log(ce::K());
  }

  // TODO: this depends on the adder circuit.
  ce::CExpr comm() const override {
    return Log(ce::K()) * ce::K() + 3 * ce::K();
  }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class MsbA2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_a2b"; }

  ce::CExpr latency() const override {
    return Log(ce::K()) + 1 + 1;
  }

  ce::CExpr comm() const override {
    return ce::K() + 2 * ce::K() + ce::K() + 32;
  }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class EqualAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_aa"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs, const NdArrayRef& rhs) const override;
};

class EqualAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_ap"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class CommonTypeV : public Kernel {
  public:
   static constexpr const char* kBindName() { return "common_type_v"; }

   Kind kind() const override { return Kind::Dynamic; }

   void evaluate(KernelEvalContext* ctx) const override;
 };

} // namespace spu::mpc::fantastic4
