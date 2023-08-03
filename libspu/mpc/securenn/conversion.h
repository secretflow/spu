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

namespace spu::mpc::securenn {

class A2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2b";

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

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class B2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

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

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class B2A_Randbit : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override {
    return ce::K() * (ce::N() - 1)  // Open bit masked value
        ;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class Msb_a2b : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2b";
  // static constexpr char kBindName[] = "msb_a2b_nosc";

  ce::CExpr latency() const override {
    return ce::Const(5)           // msb_a2a
           + (Log(ce::K()) + 1)   // adder-circuit;
                 * Log(ce::N());  // tree-reduce parties.;
  }
  ce::CExpr comm() const override {
    const auto log_p =
        9;  // in fact, now the element is ring2k_t rather than [0, p-1]
    return (9 * ce::K() + 3 * ce::K() * log_p)  // msb_a2a
           + (2 * Log(ce::K()) + 1)             // KS-adder-circuit
                 * 2 * ce::K() * (ce::N() - 1)  // And gate, for nPC
                 * (ce::N() - 1);  // (no-matter tree or ring) reduce
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

}  // namespace spu::mpc::securenn
