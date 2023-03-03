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

#include "libspu/mpc/ref2k/ref2k.h"

#include <mutex>

#include "libspu/core/trace.h"
#include "libspu/core/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

class Ref2kSecrTy : public TypeImpl<Ref2kSecrTy, RingTy, Secret> {
  using Base = TypeImpl<Ref2kSecrTy, RingTy, Secret>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "ref2k.Sec"; }
  explicit Ref2kSecrTy(FieldType field) { field_ = field; }
};

void registerTypes() {
  regPub2kTypes();

  static std::once_flag flag;
  std::call_once(
      flag, []() { TypeContext::getTypeContext()->addTypes<Ref2kSecrTy>(); });
}

class Ref2kCommonTypeS : public Kernel {
 public:
  static constexpr char kBindName[] = "common_type_s";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const Type& lhs = ctx->getParam<Type>(0);
    const Type& rhs = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.isa<Ref2kSecrTy>(), "invalid type, got={}", lhs);
    SPU_ENFORCE(rhs.isa<Ref2kSecrTy>(), "invalid type, got={}", rhs);
    ctx->setOutput(lhs);
  }
};

class Ref2kCastTypeS : public Kernel {
 public:
  static constexpr char kBindName[] = "cast_type_s";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<ArrayRef>(0);
    const auto& to_type = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, in, to_type);
    SPU_ENFORCE(in.eltype() == to_type,
                "semi2k always use same bshare type, lhs={}, rhs={}",
                in.eltype(), to_type);
    ctx->setOutput(in);
  }
};

class Ref2kP2S : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return in.as(makeType<Ref2kSecrTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kS2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "s2p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return in.as(makeType<Pub2kTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kRandS : public Kernel {
 public:
  static constexpr char kBindName[] = "rand_s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<size_t>(0)));
  }

  static ArrayRef proc(KernelEvalContext* ctx, size_t size) {
    SPU_TRACE_MPC_LEAF(ctx, size);
    auto* state = ctx->getState<PrgState>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();

    return ring_rshift(
        state->genPubl(field, size).as(makeType<Ref2kSecrTy>(field)), 2);
  }
};

class Ref2kNotS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_LEAF(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_not(in).as(makeType<Ref2kSecrTy>(field));
  }
};

class Ref2kAddSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ss";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAddSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_sp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_ss";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_sp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMatMulSS : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ss";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t m, size_t n,
                size_t k) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs, m, n, k).as(lhs.eltype());
  }
};

class Ref2kMatMulSP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_sp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t m, size_t n,
                size_t k) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_mmul(lhs, rhs, m, n, k).as(lhs.eltype());
  }
};

class Ref2kAndSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_ss";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAndSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_sp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_ss";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_sp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kLShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_LEAF(ctx, in, bits);
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class Ref2kRShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_LEAF(ctx, in, bits);
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class Ref2kBitrevS : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(start <= end);
    SPU_ENFORCE(end <= SizeOf(field) * 8);

    SPU_TRACE_MPC_LEAF(ctx, in, start, end);
    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class Ref2kARShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_LEAF(ctx, in, bits);
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class Ref2kTruncS : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_s";

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Nearest;
  }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_LEAF(ctx, in, bits);
    // Rounding
    // AxB = (AxB >> 14) + ((AxB >> 13) & 1);
    // See
    // https://stackoverflow.com/questions/14008330/how-do-you-multiply-two-fixed-point-numbers
    // Under certain pattern, like sum(mul(A, B)), error can accumulate in a
    // fairly significant way
    auto v1 = ring_arshift(in, bits);
    auto v2 = ring_arshift(in, bits - 1);
    ring_and_(v2, ring_ones(in.eltype().as<Ring2k>()->field(), in.numel()));
    ring_add_(v1, v2);
    return v1;
  }
};

class Ref2kMsbS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_s";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_LEAF(ctx, in);
    return ring_rshift(in, in.elsize() * 8 - 1).as(in.eltype());
  }
};

}  // namespace

std::unique_ptr<Object> makeRef2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  registerTypes();

  auto obj = std::make_unique<Object>("REF2K");

  // register random states & kernels.
  obj->addState<PrgState>();

  // add Z2k state.
  obj->addState<Z2kState>(conf.field());

  // register public kernels.
  regPub2kKernels(obj.get());

  // register compute kernels
  obj->regKernel<Ref2kCommonTypeS>();
  obj->regKernel<Ref2kCastTypeS>();
  obj->regKernel<Ref2kP2S>();
  obj->regKernel<Ref2kS2P>();
  obj->regKernel<Ref2kNotS>();
  obj->regKernel<Ref2kAddSS>();
  obj->regKernel<Ref2kAddSP>();
  obj->regKernel<Ref2kMulSS>();
  obj->regKernel<Ref2kMulSP>();
  obj->regKernel<Ref2kMatMulSS>();
  obj->regKernel<Ref2kMatMulSP>();
  obj->regKernel<Ref2kAndSS>();
  obj->regKernel<Ref2kAndSP>();
  obj->regKernel<Ref2kXorSS>();
  obj->regKernel<Ref2kXorSP>();
  obj->regKernel<Ref2kLShiftS>();
  obj->regKernel<Ref2kRShiftS>();
  obj->regKernel<Ref2kBitrevS>();
  obj->regKernel<Ref2kARShiftS>();
  obj->regKernel<Ref2kTruncS>();
  obj->regKernel<Ref2kMsbS>();
  obj->regKernel<Ref2kRandS>();

  return obj;
}

std::vector<ArrayRef> Ref2kIo::toShares(const ArrayRef& raw, Visibility vis,
                                        int owner_rank) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
              field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<ArrayRef>(world_size_, share);
  }
  SPU_ENFORCE(vis == VIS_SECRET, "expected SECRET, got {}", vis);

  // directly view the data as secret.
  const auto share = raw.as(makeType<Ref2kSecrTy>(field));
  return std::vector<ArrayRef>(world_size_, share);
}

ArrayRef Ref2kIo::fromShares(const std::vector<ArrayRef>& shares) const {
  const auto field = shares.at(0).eltype().as<Ring2k>()->field();
  // no matter Public or Secret, directly view the first share as public.
  return shares[0].as(makeType<RingTy>(field));
}

std::unique_ptr<Ref2kIo> makeRef2kIo(FieldType field, size_t npc) {
  registerTypes();
  return std::make_unique<Ref2kIo>(field, npc);
}

}  // namespace spu::mpc
