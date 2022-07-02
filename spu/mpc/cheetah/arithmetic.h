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

#include "spu/mpc/kernel.h"
#include "spu/mpc/semi2k/arithmetic.h"
#include "spu/mpc/util/cexpr.h"

namespace spu::mpc::cheetah {

using util::Const;
using util::K;
using util::Log;
using util::N;

typedef spu::mpc::semi2k::ZeroA ZeroA;

typedef spu::mpc::semi2k::P2A P2A;

typedef spu::mpc::semi2k::A2P A2P;

typedef spu::mpc::semi2k::NotA NotA;

typedef spu::mpc::semi2k::AddAP AddAP;

typedef spu::mpc::semi2k::AddAA AddAA;

typedef spu::mpc::semi2k::MulAP MulAP;

typedef spu::mpc::semi2k::MatMulAP MatMulAP;

typedef spu::mpc::semi2k::LShiftA LShiftA;

// typedef spu::mpc::semi2k::MulAA MulAA;
// typedef spu::mpc::semi2k::MatMulAA MatMulAA;

class TruncPrA : public TruncPrAKernel {
 private:
  bool heuristic = true;

 public:
  static constexpr char kBindName[] = "truncpr_a";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;

  bool isPrecise() const override { return false; }
};

class MsbA : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class MulAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_aa";

  Kind kind() const override { return Kind::kDynamic; }

  // TODO(juhou)
  util::CExpr latency() const override { return Const(0); }
  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_aa";

  Kind kind() const override { return Kind::kDynamic; }

  util::CExpr latency() const override { return Const(1); }

  util::CExpr comm() const override {
    // TODO(jint) express M, N, K
    return nullptr;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                size_t M, size_t N, size_t K) const override;
};

}  // namespace spu::mpc::cheetah
