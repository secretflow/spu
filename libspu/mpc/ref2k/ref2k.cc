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
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/standard_shape/protocol.h"
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
  regPV2kTypes();

  static std::once_flag flag;
  std::call_once(
      flag, []() { TypeContext::getTypeContext()->addTypes<Ref2kSecrTy>(); });
}

class Ref2kCommonTypeS : public Kernel {
 public:
  static constexpr const char* kBindName() { return "common_type_s"; }

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const Type& lhs = ctx->getParam<Type>(0);
    const Type& rhs = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.isa<Ref2kSecrTy>(), "invalid type, got={}", lhs);
    SPU_ENFORCE(rhs.isa<Ref2kSecrTy>(), "invalid type, got={}", rhs);
    ctx->pushOutput(lhs);
  }
};

class Ref2kCommonTypeV : public Kernel {
 public:
  static constexpr const char* kBindName() { return "common_type_v"; }

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const Type& lhs = ctx->getParam<Type>(0);
    const Type& rhs = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    SPU_ENFORCE(lhs.isa<Ref2kSecrTy>(), "invalid type, got={}", lhs);
    SPU_ENFORCE(rhs.isa<Ref2kSecrTy>(), "invalid type, got={}", rhs);

    const auto* lhs_v = lhs.as<Priv2kTy>();
    const auto* rhs_v = rhs.as<Priv2kTy>();

    ctx->pushOutput(
        makeType<Ref2kSecrTy>(std::max(lhs_v->field(), rhs_v->field())));
  }
};

class Ref2kCastTypeS : public CastTypeKernel {
 public:
  static constexpr const char* kBindName() { return "cast_type_s"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Type& to_type) const override {
    SPU_TRACE_MPC_DISP(ctx, in, to_type);
    SPU_ENFORCE(in.eltype() == to_type,
                "semi2k always use same bshare type, lhs={}, rhs={}",
                in.eltype(), to_type);
    return in;
  }
};

class Ref2kP2S : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in) const override {
    return in.as(makeType<Ref2kSecrTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kS2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "s2p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in) const override {
    return in.as(makeType<Pub2kTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kS2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "s2v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto field = in.eltype().as<Ring2k>()->field();
    const auto out_ty = makeType<Priv2kTy>(field, rank);
    if (comm->getRank() == rank) {  // owner
      return in.as(out_ty);
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  }
};

class Ref2kV2S : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2s"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto field = in.eltype().as<Ring2k>()->field();
    const size_t owner = in.eltype().as<Priv2kTy>()->owner();

    const auto out_ty = makeType<Ref2kSecrTy>(field);
    NdArrayRef out(out_ty, in.shape());

    int64_t numel = in.numel();

    DISPATCH_ALL_FIELDS(field, [&]() {
      std::vector<ring2k_t> _send(numel);
      NdArrayView<ring2k_t> _in(in);

      for (int64_t idx = 0; idx < numel; ++idx) {
        _send[idx] = _in[idx];
      }
      std::vector<ring2k_t> _recv = comm->bcast<ring2k_t>(_send, owner, "v2s");

      NdArrayView<ring2k_t> _out(out);
      for (int64_t idx = 0; idx < numel; ++idx) {
        _out[idx] = _recv[idx];
      }
    });
    return out;
  }
};

class Ref2kRandS : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override {
    auto* state = ctx->getState<PrgState>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();

    return ring_rshift(
        state->genPubl(field, shape).as(makeType<Ref2kSecrTy>(field)), {2});
  }
};

class Ref2kNegateS : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_neg(in).as(makeType<Ref2kSecrTy>(field));
  }
};

class Ref2kAddSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAddSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMatMulSS : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMatMulSP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    return ring_mmul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAndSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAndSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kLShiftS : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class Ref2kRShiftS : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "rshift_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class Ref2kBitrevS : public BitrevKernel {
 public:
  static constexpr const char* kBindName() { return "bitrev_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t start,
                  size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(start <= end);
    SPU_ENFORCE(end <= SizeOf(field) * 8);

    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class Ref2kARShiftS : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "arshift_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class Ref2kTruncS : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_s"; }

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Nearest;
  }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType) const override {
    // Rounding
    // AxB = (AxB >> 14) + ((AxB >> 13) & 1);
    // See
    // https://stackoverflow.com/questions/14008330/how-do-you-multiply-two-fixed-point-numbers
    // Under certain pattern, like sum(mul(A, B)), error can accumulate in a
    // fairly significant way
    auto v1 = ring_arshift(in, {static_cast<int64_t>(bits)});
    auto v2 = ring_arshift(in, {static_cast<int64_t>(bits - 1)});
    ring_and_(v2, ring_ones(in.eltype().as<Ring2k>()->field(), in.shape()));
    ring_add_(v1, v2);
    return v1;
  }
};

class Ref2kMsbS : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    return ring_rshift(in, {static_cast<int64_t>(in.elsize() * 8 - 1)})
        .as(in.eltype());
  }
};

}  // namespace

void regRef2kProtocol(SPUContext* ctx,
                      const std::shared_ptr<yacl::link::Context>& lctx) {
  registerTypes();

  // register random states & kernels.
  ctx->prot()->addState<PrgState>();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register compute kernels
  ctx->prot()
      ->regKernel<Ref2kCommonTypeS, Ref2kCommonTypeV, Ref2kCastTypeS,  //
                  Ref2kP2S, Ref2kS2P, Ref2kV2S, Ref2kS2V,              //
                  Ref2kNegateS,                                        //
                  Ref2kAddSS, Ref2kAddSP,                              //
                  Ref2kMulSS, Ref2kMulSP,                              //
                  Ref2kMatMulSS, Ref2kMatMulSP,                        //
                  Ref2kAndSS, Ref2kAndSP,                              //
                  Ref2kXorSS, Ref2kXorSP,                              //
                  Ref2kLShiftS, Ref2kRShiftS, Ref2kARShiftS,           //
                  Ref2kBitrevS,                                        //
                  Ref2kTruncS,                                         //
                  Ref2kMsbS, Ref2kRandS>();
}

std::unique_ptr<SPUContext> makeRef2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regRef2kProtocol(ctx.get(), lctx);

  return ctx;
}

Type Ref2kIo::getShareType(Visibility vis, int /*owner_rank*/) const {
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(field_);
  } else if (vis == VIS_SECRET) {
    return makeType<Ref2kSecrTy>(field_);
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<NdArrayRef> Ref2kIo::toShares(const NdArrayRef& raw, Visibility vis,
                                          int /*owner_rank*/) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
              field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<NdArrayRef>(world_size_, share);
  }
  SPU_ENFORCE(vis == VIS_SECRET, "expected SECRET, got {}", vis);

  // directly view the data as secret.
  const auto share = raw.as(makeType<Ref2kSecrTy>(field));
  return std::vector<NdArrayRef>(world_size_, share);
}

NdArrayRef Ref2kIo::fromShares(const std::vector<NdArrayRef>& shares) const {
  const auto field = shares.at(0).eltype().as<Ring2k>()->field();
  // no matter Public or Secret, directly view the first share as public.
  return shares[0].as(makeType<RingTy>(field));
}

std::unique_ptr<Ref2kIo> makeRef2kIo(FieldType field, size_t npc) {
  registerTypes();
  return std::make_unique<Ref2kIo>(field, npc);
}

}  // namespace spu::mpc
