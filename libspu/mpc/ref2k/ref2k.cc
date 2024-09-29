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
#include "libspu/mpc/generic/protocol.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

class Ref2kSecrTy : public TypeImpl<Ref2kSecrTy, ArithShare, RingTy, Secret> {
  using Base = TypeImpl<Ref2kSecrTy, ArithShare, RingTy, Secret>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "ref2k.Sec"; }
  explicit Ref2kSecrTy(SemanticType semantic_type) {
    semantic_type_ = semantic_type;
    valid_bits_ = SizeOf(semantic_type_) * 8;
    storage_type_ = GetStorageType(valid_bits_);
  }
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
    ctx->pushOutput(makeType<Ref2kSecrTy>(
        std::max(lhs.semantic_type(), rhs.semantic_type())));
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
    SPU_ENFORCE(lhs.isa<Priv2kTy>(), "invalid type, got={}", lhs);
    SPU_ENFORCE(rhs.isa<Priv2kTy>(), "invalid type, got={}", rhs);

    const auto* lhs_v = lhs.as<Priv2kTy>();
    const auto* rhs_v = rhs.as<Priv2kTy>();

    SPU_ENFORCE(lhs_v->semantic_type() == rhs_v->semantic_type(),
                "semantic type mismatch");

    ctx->pushOutput(makeType<Ref2kSecrTy>(lhs_v->semantic_type()));
  }
};

class Ref2kCastTypeS : public CastTypeKernel {
 public:
  static constexpr const char* kBindName() { return "cast_type_s"; }

  Kind kind() const override { return Kind::Dynamic; }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Type& to_type) const override {
    SPU_TRACE_MPC_DISP(ctx, in, to_type);
    MemRef out(to_type, in.shape());
    ring_assign(out, in);
    return out;
  }
};

class Ref2kP2S : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in) const override {
    const auto* in_t = in.eltype().as<Pub2kTy>();
    return in.as(makeType<Ref2kSecrTy>(in_t->semantic_type()));
  }
};

class Ref2kS2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "s2p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in) const override {
    const auto* in_t = in.eltype().as<Ref2kSecrTy>();
    return in.as(makeType<Pub2kTy>(in_t->semantic_type()));
  }
};

class Ref2kS2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "s2v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              size_t rank) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto* in_t = in.eltype().as<Ref2kSecrTy>();
    const auto out_t = makeType<Priv2kTy>(in_t->semantic_type(), rank);
    if (comm->getRank() == rank) {  // owner
      return in.as(out_t);
    } else {
      return makeConstantArrayRef(out_t, in.shape());
    }
  }
};

class Ref2kV2S : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2s"; }

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto* in_t = in.eltype().as<Priv2kTy>();
    auto owner = in_t->owner();

    const auto out_t = makeType<Ref2kSecrTy>(in_t->semantic_type());
    MemRef out(out_t, in.shape());

    int64_t numel = in.numel();

    DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
      std::vector<ScalarT> _send(numel);
      MemRefView<ScalarT> _in(in);

      for (int64_t idx = 0; idx < numel; ++idx) {
        _send[idx] = _in[idx];
      }
      std::vector<ScalarT> _recv = comm->bcast<ScalarT>(_send, owner, "v2s");

      MemRefView<ScalarT> _out(out);
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

  MemRef proc(KernelEvalContext* ctx, SemanticType type,
              const Shape& shape) const override {
    auto* state = ctx->getState<PrgState>();
    auto out_t = makeType<Ref2kSecrTy>(type);

    MemRef out(out_t, shape);

    state->fillPubl(out.data<std::byte>(), out.numel() * out.elsize());

    ring_rshift_(out, {2});

    return out;
  }
};

class Ref2kNegateS : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override {
    return ring_neg(in).as(in.eltype());
  }
};

class Ref2kAddSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    return ring_add(lhs, rhs);
  }
};

class Ref2kAddSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    return ring_add(lhs, rhs);
  }
};

class Ref2kMulSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    auto ret = ring_mul(lhs, rhs);
    return ret.as(makeType<Ref2kSecrTy>(ret.eltype().semantic_type()));
  }
};

class Ref2kMulSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    auto ret = ring_mul(lhs, rhs);
    return ret.as(makeType<Ref2kSecrTy>(ret.eltype().semantic_type()));
  }
};

class Ref2kMatMulSS : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    auto ret = ring_mmul(lhs, rhs);
    return ret.as(makeType<Ref2kSecrTy>(ret.eltype().semantic_type()));
  }
};

class Ref2kMatMulSP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    auto ret = ring_mmul(lhs, rhs);
    return ret.as(makeType<Ref2kSecrTy>(ret.eltype().semantic_type()));
  }
};

class Ref2kAndSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs);
  }
};

class Ref2kAndSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    return ring_and(lhs, rhs);
  }
};

class Ref2kXorSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_ss"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype(), "lhs = {}, rhs = {}",
                lhs.eltype(), rhs.eltype());
    return ring_xor(lhs, rhs);
  }
};

class Ref2kXorSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_sp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    return ring_xor(lhs, rhs);
  }
};

class Ref2kLShiftS : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Sizes& bits) const override {
    return ring_lshift(in, bits);
  }
};

class Ref2kRShiftS : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "rshift_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Sizes& bits) const override {
    return ring_rshift(in, bits);
  }
};

class Ref2kBitrevS : public BitrevKernel {
 public:
  static constexpr const char* kBindName() { return "bitrev_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in, size_t start,
              size_t end) const override {
    return ring_bitrev(in, start, end);
  }
};

class Ref2kARShiftS : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "arshift_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Sizes& bits) const override {
    return ring_arshift(in, bits);
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

  MemRef proc(KernelEvalContext* ctx, const MemRef& in, size_t bits,
              SignType) const override {
    // Rounding
    // AxB = (AxB >> 14) + ((AxB >> 13) & 1);
    // See
    // https://stackoverflow.com/questions/14008330/how-do-you-multiply-two-fixed-point-numbers
    // Under certain pattern, like sum(mul(A, B)), error can accumulate in a
    // fairly significant way
    auto v1 = ring_arshift(in, {static_cast<int64_t>(bits)});
    auto v2 = ring_arshift(in, {static_cast<int64_t>(bits - 1)});
    MemRef one(in.eltype(), in.shape());
    ring_ones(one);
    ring_and_(v2, one);
    ring_add_(v1, v2);
    return v1;
  }
};

class Ref2kMsbS : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override {
    MemRef res(makeType<Ref2kSecrTy>(SE_1), in.shape());
    ring_msb(res, in);
    return res;
  }
};

namespace {

template <typename InT, typename OutT>
void copyRing(MemRef& out, const MemRef& in) {
  MemRefView<InT> in_view(in);
  MemRefView<OutT> out_view(out);

  auto numel = out.numel();

  for (int64_t idx = 0; idx < numel; ++idx) {
    out_view[idx] = in_view[idx];
  }
}

}  // namespace

class Ref2kRingCastS : public RingCastKernel {
 public:
  static constexpr const char* kBindName() { return "ring_cast_s"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in,
              SemanticType to_type) const override {
    MemRef out(makeType<Ref2kSecrTy>(to_type), in.shape());

    bool inUnsigned = isUnsigned(in.eltype().semantic_type());
    bool outUnsigned = isUnsigned(out.eltype().semantic_type());

    DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
      using InT = ScalarT;
      DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
        using OutT = ScalarT;
        if (inUnsigned && outUnsigned) {
          copyRing<InT, OutT>(out, in);
        } else if (!inUnsigned && !outUnsigned) {
          copyRing<std::make_signed_t<InT>, std::make_signed_t<OutT>>(out, in);
        } else if (!inUnsigned) {
          copyRing<std::make_signed_t<InT>, OutT>(out, in);
        } else {
          copyRing<InT, std::make_signed_t<OutT>>(out, in);
        }
      });
    });

    return out;
  }
};

class Ref2kEqualSS : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_aa"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& x,
              const MemRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());
    MemRef ret(makeType<Ref2kSecrTy>(SE_1), x.shape());
    ring_equal(ret, x, y);
    return ret;
  }
};

class Ref2kEqualSP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& x,
              const MemRef& y) const override {
    MemRef ret(makeType<Ref2kSecrTy>(SE_1), x.shape());
    ring_equal(ret, x, y);
    return ret;
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
  ctx->prot()->addState<Z2kState>(ctx->config().protocol().field());

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regGenericKernels(ctx);

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
                  Ref2kEqualSP, Ref2kEqualSS,                          //
                  Ref2kMsbS, Ref2kRingCastS, Ref2kRandS>();
}

std::unique_ptr<SPUContext> makeRef2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regRef2kProtocol(ctx.get(), lctx);

  return ctx;
}

Type Ref2kIo::getShareType(Visibility vis, PtType type,
                           int /*owner_rank*/) const {
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(GetEncodedType(type, field_));
  } else if (vis == VIS_SECRET) {
    return makeType<Ref2kSecrTy>(GetEncodedType(type, field_));
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<MemRef> Ref2kIo::toShares(const MemRef& raw, Visibility vis,
                                      int /*owner_rank*/) const {
  const auto* in_t = raw.eltype().as<RingTy>();
  SPU_ENFORCE(in_t, "expected RingTy, got {}", raw.eltype());

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(in_t->semantic_type()));
    return std::vector<MemRef>(world_size_, share);
  }
  SPU_ENFORCE(vis == VIS_SECRET, "expected SECRET, got {}", vis);

  // directly view the data as secret.
  const auto share = raw.as(makeType<Ref2kSecrTy>(in_t->semantic_type()));
  return std::vector<MemRef>(world_size_, share);
}

MemRef Ref2kIo::fromShares(const std::vector<MemRef>& shares) const {
  const auto* in_t = shares.at(0).eltype().as<RingTy>();
  // no matter Public or Secret, directly view the first share as public.
  return shares[0].as(
      makeType<RingTy>(in_t->semantic_type(), in_t->size() * 8));
}

std::unique_ptr<Ref2kIo> makeRef2kIo(size_t field, size_t npc) {
  registerTypes();
  return std::make_unique<Ref2kIo>(field, npc);
}

}  // namespace spu::mpc
