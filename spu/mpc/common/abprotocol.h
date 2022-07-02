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

#include "spu/mpc/object.h"

namespace spu::mpc {

class ABProtState : public State {
 public:
  static constexpr char kBindName[] = "ABProtState";

  bool lazy_ab = true;
};

class ABProtP2S : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2s";
  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class ABProtS2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "s2p";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class ABProtNotS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class ABProtAddSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtAddSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtMulSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtMulSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtMatMulSP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                size_t M, size_t N, size_t K) const override;
};

class ABProtMatMulSS : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                size_t M, size_t N, size_t K) const override;
};

class ABProtAndSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtAndSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtXorSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtXorSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class ABProtEqzS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "eqz_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class ABProtLShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ABProtRShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ABProtARShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ABProtTruncPrS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "truncpr_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};

class ABProtBitrevS : public Kernel {
 public:
  static constexpr char kBindName[] = "bitrev_s";

  Kind kind() const override { return Kind::kDynamic; }

  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<size_t>(1), ctx->getParam<size_t>(2)));
  }
  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const;
};

class ABProtMsbS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

}  // namespace spu::mpc
