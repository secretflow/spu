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

namespace spu::mpc::cheetah {

class A2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& x) const override;
};

class B2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& x) const override;
};

class CommonTypeV : public Kernel {
 public:
  static constexpr const char* kBindName() { return "common_type_v"; }

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override;
};

class RingCastS : public RingCastKernel {
 public:
  static constexpr const char* kBindName() { return "ring_cast_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in,
              SemanticType to_type) const override;
};

class BitDecompose : public DecomposeKernel {
 public:
  static constexpr const char* kBindName() { return "bit_decompose_b"; }

  ce::CExpr latency() const override { return 0; }

  ce::CExpr comm() const override { return 0; }

  std::vector<MemRef> proc(KernelEvalContext* ctx,
                           const MemRef& in) const override;
};

class BitCompose : public ComposeKernel {
 public:
  static constexpr const char* kBindName() { return "bit_compose_b"; }

  ce::CExpr latency() const override { return 0; }

  ce::CExpr comm() const override { return 0; }

  MemRef proc(KernelEvalContext* ctx,
              const std::vector<MemRef>& in) const override;
};

}  // namespace spu::mpc::cheetah
