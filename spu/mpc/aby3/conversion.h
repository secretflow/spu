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

#include "spu/core/array_ref.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/kernel.h"

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
// TODO(junfeng): Optimize anount of comm.
class A2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2b";

  CExpr latency() const override {
    // 1 * AddBB : log(k) + 1
    // 1 * rotate: 1
    return Log(K()) + 1 + 1;
  }

  CExpr comm() const override {
    // 1 * AddBB : logk * 2k + k
    // 1 * rotate: k
    return Log(K()) * K() * 2 + K() * 2;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

// Referrence:
// IV.E Boolean to Arithmetic Sharing (B2A), extended to 3pc settings.
// https://encrypto.de/papers/DSZ15.pdf
//
// Latency: 4 + log(nbits) - 3 rotate + 1 send/rec + 1 ppa.
// TODO(junfeng): Optimize anount of comm.
class B2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  CExpr latency() const override {
    // 3 * rotate   : 3
    // 1 * AddBB    : 1 + logk
    // manual set   : 1
    return Const(5) + Log(K());
  }

  CExpr comm() const override {
    // 3 * rotate   : 3k
    // 1 * AddBB    : logk * 2k + k
    // manual add   : k
    return Log(K()) * K() * 2 + K() * 5;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2.
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

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class AddBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_bb";

  CExpr latency() const override {
    // Cost from other gates (from KoggleStoneAdder):
    // 1 * AddBB    : 1
    // logk * AndBB : 2logk (if vectorize, logk)
    return Log(K()) + Const(1);
  }

  CExpr comm() const override {
    // Cost from other gates (from KoggleStoneAdder):
    // 1 * AddBB    : k
    // logk * AndBB : logk * 2k
    return Log(K()) * K() * 2 + K();
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

}  // namespace spu::mpc::aby3
