// Copyright 2025 Ant Group Co., Ltd.
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

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::fantastic4 {

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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Type& to_type) const override;
};

class B2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2p"; }

  ce::CExpr latency() const override {
    // rotate : 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // rotate : k
    return ce::K();
  }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class P2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class B2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "b2v"; }

  ce::CExpr latency() const override {
    // 1 * send/recv: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override;
};

class AndBP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_bp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class AndBB : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_bb"; }

  ce::CExpr latency() const override {
    // rotate : 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // rotate : k
    return ce::K();
  }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class XorBP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_bp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class XorBB : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_bb"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class LShiftB : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override;
};

class RShiftB : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "rshift_b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override;
};

class ARShiftB : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "arshift_b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override;
};

class BitrevB : public BitrevKernel {
 public:
  static constexpr const char* kBindName() { return "bitrev_b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t start,
                  size_t end) const override;
};

class BitIntlB : public BitSplitKernel {
 public:
  static constexpr const char* kBindName() { return "bitintl_b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t stride) const override;
};

class BitDeintlB : public BitSplitKernel {
 public:
  static constexpr const char* kBindName() { return "bitdeintl_b"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t stride) const override;
};

}  // namespace spu::mpc::fantastic4
