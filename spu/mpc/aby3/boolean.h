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

class B2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2p";

  CExpr latency() const override {
    // rotate : 1
    return Const(1);
  }

  CExpr comm() const override {
    // rotate : k
    return K();
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class P2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class AndBP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_bp";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AndBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_bb";

  CExpr latency() const override {
    // rotate : 1
    return Const(1);
  }

  CExpr comm() const override {
    // rotate : k
    return K();
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class XorBP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_bp";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class XorBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_bb";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class LShiftB : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class RShiftB : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ARShiftB : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class BitrevB : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override;
};

}  // namespace spu::mpc::aby3
