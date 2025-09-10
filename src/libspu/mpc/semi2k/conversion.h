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

namespace spu::mpc::semi2k {

class A2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2b"; }

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

class A2B_Bits : public SharingConvertKernel {
 public:
  static constexpr const char* kBindName() { return "a2b_bits"; }

  // the exact costs depends on the nbits and adder circuit.
  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  int64_t nbits) const override;
};

class B2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }

  ce::CExpr latency() const override {
    return (Log(ce::K()) + 1) * Log(ce::N())  // A2B
           + Log(ce::K() + 1)                 // add_bb
           + 1                                // reveal
        ;
  }

  ce::CExpr comm() const override {
    const auto n_1 = ce::N() - 1;
    return (2 * Log(ce::K()) + 1) * 2 * ce::K() * n_1 * n_1  // A2B
           + (2 * Log(ce::K()) + 1) * 2 * ce::K()            // add_bb
        ;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

class B2A_Randbit : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override {
    return ce::K() * (ce::N() - 1)  // Open bit masked value
        ;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

class B2A_Disassemble : public DisassembleKernel {
 public:
  static constexpr const char* kBindName() { return "b2a_disassemble"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override {
    return ce::K() * (ce::N() - 1)  // Open bit masked value
        ;
  }

  std::vector<NdArrayRef> proc(KernelEvalContext* ctx,
                               const NdArrayRef& x) const override;
};

// Note: current only for 2PC.
class MsbA2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_a2b"; }

  ce::CExpr latency() const override {
    // 1 * carry: log(k) + 1
    return Log(ce::K()) + 1;
  }

  ce::CExpr comm() const override {
    // 1 * and_bb: 2 * k * (N-1)
    // 1 * carrya2b: k + k/2 + k/4 + ... + 8 (8) + 8 (4) + 8 (2) + 8 (1)
    return 2 * ce::K() * (ce::N() - 1) + 2 * (ce::N() - 1) * (2 * ce::K() + 32);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class EqualAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_aa"; }

  ce::CExpr latency() const override {
    // 1 * edabits + logk * andbb
    return Log(ce::K()) + 1;
  }

  ce::CExpr comm() const override {
    return (2 * Log(ce::K()) + 1) * ce::K() * (ce::N() - 1);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class EqualAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_ap"; }

  ce::CExpr latency() const override {
    // 1 * edabits + logk * andbb
    return Log(ce::K()) + 1;
  }

  ce::CExpr comm() const override {
    return (2 * Log(ce::K()) + 1) * ce::K() * (ce::N() - 1);
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class CommonTypeV : public Kernel {
 public:
  static constexpr const char* kBindName() { return "common_type_v"; }

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override;
};

}  // namespace spu::mpc::semi2k
