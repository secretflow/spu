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
#include "libspu/mpc/semi2k/arithmetic.h"
#include "libspu/mpc/utils/cexpr.h"

namespace spu::mpc::cheetah {

using ZeroA = spu::mpc::semi2k::ZeroA;

using RandA = spu::mpc::semi2k::RandA;

using P2A = spu::mpc::semi2k::P2A;

using A2P = spu::mpc::semi2k::A2P;

using NotA = spu::mpc::semi2k::NotA;

using AddAP = spu::mpc::semi2k::AddAP;

using AddAA = spu::mpc::semi2k::AddAA;

using MulAP = spu::mpc::semi2k::MulAP;

using MatMulAP = spu::mpc::semi2k::MatMulAP;

using LShiftA = spu::mpc::semi2k::LShiftA;

class TruncA : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_a";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                size_t bits) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

class MsbA2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2b";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class MulA1B : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_a1b";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return 2 * ce::K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& ashr,
                const ArrayRef& bshr) const override;
};

class MulAA : public BinaryKernel {
 private:
  ArrayRef mulDirectly(KernelEvalContext* ctx, const ArrayRef& lhs,
                       const ArrayRef& rhs) const;

  ArrayRef mulWithBeaver(KernelEvalContext* ctx, const ArrayRef& lhs,
                         const ArrayRef& rhs) const;

 public:
  static constexpr char kBindName[] = "mul_aa";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_aa";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override {
    // TODO(jint) express M, N, K
    return nullptr;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x, const ArrayRef& y,
                size_t m, size_t n, size_t k) const override;
};

}  // namespace spu::mpc::cheetah
