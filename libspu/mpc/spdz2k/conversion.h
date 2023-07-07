// Copyright 2023 Ant Group Co., Ltd.
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
#include "libspu/core/cexpr.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/spdz2k/value.h"

namespace spu::mpc::spdz2k {

using ce::CExpr;
using ce::Const;
using ce::K;
using ce::Log;
using ce::N;

class A2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2b";

  CExpr latency() const override {
    // 1 * AddBB : log(k) + 1
    // 1 * rotate: 1
    // return Log(K()) + 1 + 1;
    return Const(0);
  }

  CExpr comm() const override {
    // 1 * AddBB : logk * 2k + k
    // 1 * rotate: k
    // return Log(K()) * K() * 2 + K() * 2;
    return Const(0);
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class A2Bit : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2bit";

  CExpr latency() const override {
    // 1 * AddBB : log(k) + 1
    // 1 * rotate: 1
    // return Log(K()) + 1 + 1;
    return Const(0);
  }

  CExpr comm() const override {
    // 1 * AddBB : logk * 2k + k
    // 1 * rotate: k
    // return Log(K()) * K() * 2 + K() * 2;
    return Const(0);
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class Bit2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "bit2a";

  CExpr latency() const override {
    // 1 * AddBB : log(k) + 1
    // 1 * rotate: 1
    // return Log(K()) + 1 + 1;
    return Const(0);
  }

  CExpr comm() const override {
    // 1 * AddBB : logk * 2k + k
    // 1 * rotate: k
    // return Log(K()) * K() * 2 + K() * 2;
    return Const(0);
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class BitDec : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "bit_dec";

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
class B2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  CExpr latency() const override {
    // 2 * rotate   : 2
    // 1 * AddBB    : 1 + logk
    // return Const(3) + Log(K());
    return Const(0);
  }

  CExpr comm() const override {
    // 2 * rotate   : 2k
    // 1 * AddBB    : logk * 2k + k
    // return Log(K()) * K() * 2 + 3 * K();
    return Const(0);
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class MSB : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2b";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class AddBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_bb";

  CExpr latency() const override {
    // Cost from other gates (from KoggeStoneAdder):
    // 1 * AddBB    : 1
    // logk * AndBB : 2logk (if vectorize, logk)
    // return Log(K()) + Const(1);
    return Const(0);
  }

  CExpr comm() const override {
    // Cost from other gates (from KoggeStoneAdder):
    // 1 * AddBB    : k
    // logk * AndBB : logk * 2k
    // return Log(K()) * K() * 2 + K();
    return Const(0);
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AddBP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_bp";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override;
};

class BitLTBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "bitlt_bb";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override;
};

class BitLEBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "bitle_bb";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override;
};

}  // namespace spu::mpc::spdz2k