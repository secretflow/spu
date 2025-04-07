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

namespace spu::mpc::swift {

class A2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2b"; }

  ce::CExpr latency() const override {
    return (Log(ce::K()) + 1)  // adder-circuit
           * 13                // And gate
           * 2                 // 2 calls of circuit
        ;
  }

  ce::CExpr comm() const override {
    return (2 * Log(ce::K()) + 1)  // KS-adder-circuit
           * ce::K() * 7           // And gate
           * 2                     // 2 calls of circuit
        ;
  }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class MsbA2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_a2b"; }

  ce::CExpr latency() const override {
    return (Log(ce::K()) + 1)  // adder-circuit
           * 13                // And gate
           * 2                 // 2 calls of circuit
        ;
  }

  ce::CExpr comm() const override {
    return (2 * Log(ce::K()) + 1)  // KS-adder-circuit
           * ce::K() * 7           // And gate
           * 2                     // 2 calls of circuit
        ;
  }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class B2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }

  ce::CExpr latency() const override { return ce::Const(13 * 2); }

  ce::CExpr comm() const override {
    // 2 * mult
    // Sizeof(Field) * (comm of MulAA) * 2
    return ce::K() * ce::K() * 10 * 2;
  }

  // The comm is incorrect for FM128, since SWIFT only support FM32 and FM64
  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

}  // namespace spu::mpc::swift
