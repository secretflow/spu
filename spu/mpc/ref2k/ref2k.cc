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

#include "spu/mpc/ref2k/ref2k.h"

#include <mutex>

#include "spu/core/type.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/util/ring_ops.h"

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

  Kind kind() const override { return Kind::kDynamic; }

  void evaluate(EvalContext* ctx) const override {
    const Type& lhs = ctx->getParam<Type>(0);
    const Type& rhs = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    YACL_ENFORCE(lhs.isa<Ref2kSecrTy>(), "invalid type, got={}", lhs);
    YACL_ENFORCE(rhs.isa<Ref2kSecrTy>(), "invalid type, got={}", rhs);
    ctx->setOutput(lhs);
  }
};

class Ref2kCastTypeS : public Kernel {
 public:
  static constexpr char kBindName[] = "cast_type_s";

  Kind kind() const override { return Kind::kDynamic; }

  void evaluate(EvalContext* ctx) const override {
    const auto& in = ctx->getParam<ArrayRef>(0);
    const auto& to_type = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, in, to_type);
    YACL_ENFORCE(in.eltype() == to_type,
                 "semi2k always use same bshare type, lhs={}, rhs={}",
                 in.eltype(), to_type);
    ctx->setOutput(in);
  }
};

class Ref2kP2S : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return in.as(makeType<Ref2kSecrTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kS2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "s2p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return in.as(makeType<Pub2kTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kRandS : public Kernel {
 public:
  static constexpr char kBindName[] = "rand_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<size_t>(0)));
  }

  ArrayRef proc(KernelEvalContext* ctx, size_t size) const {
    SPU_TRACE_MPC_LEAF(ctx, size);
    auto* state = ctx->caller()->getState<PrgState>();
    const auto field = ctx->caller()->getState<Z2kState>()->getDefaultField();

    return ring_rshift(
        state->genPubl(field, size).as(makeType<Ref2kSecrTy>(field)), 2);
  }
};

class Ref2kNotS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_LEAF(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_not(in).as(makeType<Ref2kSecrTy>(field));
  }
};

class Ref2kEqzS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "eqz_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_LEAF(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_equal(in, ring_zeros(field, in.numel())).as(in.eltype());
  }
};

class Ref2kAddSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    YACL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAddSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    YACL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMatMulSS : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t M, size_t N,
                size_t K) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    YACL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs, M, N, K).as(lhs.eltype());
  }
};

class Ref2kMatMulSP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t M, size_t N,
                size_t K) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_mmul(lhs, rhs, M, N, K).as(lhs.eltype());
  }
};

class Ref2kAndSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    YACL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAndSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    YACL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kLShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_LEAF(ctx, in, bits);
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class Ref2kRShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_LEAF(ctx, in, bits);
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class Ref2kBitrevS : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    YACL_ENFORCE(start <= end);
    YACL_ENFORCE(end <= SizeOf(field) * 8);

    SPU_TRACE_MPC_LEAF(ctx, in, start, end);
    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class Ref2kARShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_LEAF(ctx, in, bits);
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class Ref2kMsbS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

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
  obj->regKernel<Ref2kEqzS>();
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
  obj->regKernel<Ref2kARShiftS>("truncpr_s");
  obj->regKernel<Ref2kMsbS>();
  obj->regKernel<Ref2kRandS>();

  return obj;
}

std::vector<ArrayRef> Ref2kIo::toShares(const ArrayRef& raw, Visibility vis,
                                        int owner_rank) const {
  YACL_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
               raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  YACL_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
               field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<ArrayRef>(world_size_, share);
  }
  YACL_ENFORCE(vis == VIS_SECRET, "expected SECRET, got {}", vis);

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
