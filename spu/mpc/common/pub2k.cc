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

#include "spu/mpc/common/pub2k.h"

#include <mutex>

#include "spu/core/array_ref.h"
#include "spu/core/profile.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/kernel.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {

class Pub2kRandP : public Kernel {
 public:
  static constexpr char kBindName[] = "rand_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<FieldType>(0), ctx->getParam<size_t>(1)));
  }

  ArrayRef proc(KernelEvalContext* ctx, FieldType field, size_t size) const {
    SPU_PROFILE_TRACE_KERNEL(ctx, size);
    auto* state = ctx->caller()->getState<PrgState>();
    return state->genPubl(field, size).as(makeType<Pub2kTy>(field));
  }
};

class Pub2kNotP : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_not(in).as(makeType<Pub2kTy>(field));
  }
};

class Pub2kEqzP : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "eqz_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_equal(in, ring_zeros(field, in.numel())).as(in.eltype());
  }
};

class Pub2kAddPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_pp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kMulPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_pp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kMatMulPP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_pp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t M, size_t N,
                size_t K) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs, M, N, K).as(lhs.eltype());
  }
};

class Pub2kAndPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_pp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kXorPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_pp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kLShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class Pub2kRShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class Pub2kBitrevP : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    YASL_ENFORCE(start <= end);
    YASL_ENFORCE(end <= SizeOf(field) * 8);

    SPU_PROFILE_TRACE_KERNEL(ctx, in, start, end);
    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class Pub2kARShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class Pub2kMsbP : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in);
    return ring_rshift(in, in.elsize() * 8 - 1).as(in.eltype());
  }
};

void regPub2kTypes() {
  static std::once_flag flag;

  std::call_once(flag,
                 []() { TypeContext::getTypeContext()->addTypes<Pub2kTy>(); });
}

void regPub2kKernels(Object* obj) {
  obj->regKernel<Pub2kRandP>();
  obj->regKernel<Pub2kNotP>();
  obj->regKernel<Pub2kEqzP>();
  obj->regKernel<Pub2kAddPP>();
  obj->regKernel<Pub2kMulPP>();
  obj->regKernel<Pub2kMatMulPP>();
  obj->regKernel<Pub2kAndPP>();
  obj->regKernel<Pub2kXorPP>();
  obj->regKernel<Pub2kLShiftP>();
  obj->regKernel<Pub2kRShiftP>();
  obj->regKernel<Pub2kBitrevP>();
  obj->regKernel<Pub2kARShiftP>();
  obj->regKernel<Pub2kMsbP>();
}

}  // namespace spu::mpc
