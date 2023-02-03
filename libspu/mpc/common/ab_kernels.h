// Copyright 2022 Ant Group Co., Ltd.
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
#include "libspu/mpc/util/cexpr.h"

namespace spu::mpc::common {

using util::CExpr;
using util::Const;
using util::K;
using util::Log;
using util::N;

class BitIntlB : public Kernel {
 public:
  static constexpr char kBindName[] = "bitintl_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  void evaluate(KernelEvalContext* ctx) const override;
};

class BitDeintlB : public Kernel {
 public:
  static constexpr char kBindName[] = "bitdeintl_b";

  CExpr latency() const override { return Const(0); }

  CExpr comm() const override { return Const(0); }

  void evaluate(KernelEvalContext* ctx) const override;
};

class AddBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_bb";

  enum class CircuitType {
    KoggeStone,
    Sklansky,
    Count,
  };

 private:
  CircuitType type_;

 public:
  explicit AddBB(CircuitType type = CircuitType::KoggeStone) : type_(type) {}

  CExpr latency() const override {
    // For both sklansky/ks
    return Log(K()) + Const(1);
  }

  // TODO: depends on the and gate.
  CExpr comm() const override {
    switch (type_) {
      case CircuitType::KoggeStone: {
        // FIXME: this is aby3 cost
        // 1 * and        : k
        // logk * 2 * and : logk * 2k
        return Log(K()) * K() * 2 + K();
      }
      case CircuitType::Sklansky: {
        // FIXME: this is aby3 cost
        // 1 * and    : k
        // logk * and : logk * k
        return Log(K()) * K() + K();
      }
      default:
        YACL_THROW("should not be here");
    }
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

// compute the k'th bit of x + y
ArrayRef carry_out(Object* ctx, const ArrayRef& x, const ArrayRef& y, size_t k);

}  // namespace spu::mpc::common
