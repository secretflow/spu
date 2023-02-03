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

#include "libspu/core/array_ref.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::aby3 {

using util::CExpr;
using util::Const;
using util::K;
using util::Log;
using util::N;

// Referrence:
// ABY3: A Mixed Protocol Framework for Machine Learning
// P16 5.3 Share Conversions, Bit Decomposition
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2 + log(nbits) from 2 rotate and 1 ppa.
class A2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2b";

  CExpr latency() const override {
    // 1 * AddBB : log(k) + 1
    // 1 * rotate: 1
    return Log(K()) + 1 + 1;
  }

  // TODO: this depends on the adder circuit.
  CExpr comm() const override {
    // 1 * AddBB : logk * k + k
    // 1 * rotate: k
    return Log(K()) * K() + K() * 2;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class B2ASelector : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

// Referrence:
// IV.E Boolean to Arithmetic Sharing (B2A), extended to 3pc settings.
// https://encrypto.de/papers/DSZ15.pdf
class B2AByPPA : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  CExpr latency() const override {
    // 2 * rotate   : 2
    // 1 * AddBB    : 1 + logk
    return Const(3) + Log(K());
  }

  // TODO: this depends on the adder circuit.
  CExpr comm() const override {
    // 2 * rotate   : 2k
    // 1 * AddBB    : logk * k + k
    return Log(K()) * K() + 3 * K();
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
class B2AByOT : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  CExpr latency() const override { return Const(2); }

  // Note: when nbits is large, OT method will be slower then circuit method.
  CExpr comm() const override {
    return 2 * K() * K()  // the OT
           + K()          // partial send
        ;
  }

  // FIXME: bypass unittest.
  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class MsbA2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2b";

  util::CExpr latency() const override {
    // 1 * carry_out : log(k) + 1
    // 1 * rotate: 1
    return Log(K()) + 1 + 1;
  }

  util::CExpr comm() const override {
    // 1 * carry_out: k + 2 * k
    // 1 * rotate: k
    return K() * 4;
  }

  float getCommTolerance() const override { return 0.2; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

}  // namespace spu::mpc::aby3