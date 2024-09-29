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

#include "libspu/mpc/kernel.h"

namespace spu::mpc::securenn {

class CommonTypeB : public Kernel {
 public:
  static constexpr const char* kBindName() { return "common_type_b"; }

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override;
};

class CastTypeB : public CastTypeKernel {
 public:
  static constexpr const char* kBindName() { return "cast_type_b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Type& to_type) const override;
};

class B2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2p"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * (ce::N() - 1); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override;
};

class P2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override;
};

class AndBP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_bp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override;
};

class AndBB : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_bb"; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * 2 * (ce::N() - 1); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override;
};

class XorBP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_bp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override;
};

class XorBB : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_bb"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override;
};

}  // namespace spu::mpc::securenn
