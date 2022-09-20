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

#include "spu/mpc/common/abprotocol.h"

#include "spu/core/profile.h"
#include "spu/mpc/common/pub2k.h"

namespace spu::mpc {
namespace {

ArrayRef _Lazy2B(Object* obj, const ArrayRef& in) {
  if (in.eltype().isa<AShare>()) {
    return obj->call("a2b", in);
  } else {
    YASL_ENFORCE(in.eltype().isa<BShare>());
    return in;
  }
}

ArrayRef _Lazy2A(Object* obj, const ArrayRef& in) {
  if (in.eltype().isa<BShare>()) {
    return obj->call("b2a", in);
  } else {
    YASL_ENFORCE(in.eltype().isa<AShare>(), "expect AShare, got {}",
                 in.eltype());
    return in;
  }
}

#define _LAZY_AB ctx->caller()->getState<ABProtState>()->lazy_ab

#define _2A(x) _Lazy2A(ctx->caller(), x)
#define _2B(x) _Lazy2B(ctx->caller(), x)

#define _IsA(x) x.eltype().isa<AShare>()
#define _IsB(x) x.eltype().isa<BShare>()
#define _IsP(x) x.eltype().isa<Public>()
#define _NBits(x) x.eltype().as<BShare>()->nbits()

#define _A2P(x) ctx->caller()->call("a2p", x)
#define _P2A(x) ctx->caller()->call("p2a", x)
#define _NotA(x) ctx->caller()->call("not_a", x)
#define _AddAP(lhs, rhs) ctx->caller()->call("add_ap", lhs, rhs)
#define _AddAA(lhs, rhs) ctx->caller()->call("add_aa", lhs, rhs)
#define _MulAP(lhs, rhs) ctx->caller()->call("mul_ap", lhs, rhs)
#define _MulAA(lhs, rhs) ctx->caller()->call("mul_aa", lhs, rhs)
#define _HasMulA1B() ctx->caller()->hasKernel("mul_a1b")
#define _MulA1B(lhs, rhs) ctx->caller()->call("mul_a1b", lhs, rhs)
#define _LShiftA(in, bits) ctx->caller()->call("lshift_a", in, bits)
#define _TruncPrA(in, bits) ctx->caller()->call("truncpr_a", in, bits)
#define _MatMulAP(A, B, M, N, K) ctx->caller()->call("mmul_ap", A, B, M, N, K)
#define _MatMulAA(A, B, M, N, K) ctx->caller()->call("mmul_aa", A, B, M, N, K)
#define _B2P(x) ctx->caller()->call("b2p", x)
#define _P2B(x) ctx->caller()->call("p2b", x)
#define _A2B(x) ctx->caller()->call("a2b", x)
#define _B2A(x) ctx->caller()->call("b2a", x)
#define _NotB(x) ctx->caller()->call("not_b", x)
#define _AndBP(lhs, rhs) ctx->caller()->call("and_bp", lhs, rhs)
#define _AndBB(lhs, rhs) ctx->caller()->call("and_bb", lhs, rhs)
#define _XorBP(lhs, rhs) ctx->caller()->call("xor_bp", lhs, rhs)
#define _XorBB(lhs, rhs) ctx->caller()->call("xor_bb", lhs, rhs)
#define _LShiftB(in, bits) ctx->caller()->call("lshift_b", in, bits)
#define _RShiftB(in, bits) ctx->caller()->call("rshift_b", in, bits)
#define _ARShiftB(in, bits) ctx->caller()->call("arshift_b", in, bits)
#define _RitrevB(in, start, end) ctx->caller()->call("bitrev_b", in, start, end)
#define _MsbA(in) ctx->caller()->call("msb_a", in)

class ABProtState : public State {
 public:
  static constexpr char kBindName[] = "ABProtState";

  bool lazy_ab = true;
};

class ABProtCommonTypeS : public Kernel {
 public:
  static constexpr char kBindName[] = "common_type_s";

  Kind kind() const override { return Kind::kDynamic; }

  void evaluate(EvalContext* ctx) const override {
    const Type& lhs = ctx->getParam<Type>(0);
    const Type& rhs = ctx->getParam<Type>(1);

    SPU_TRACE_KERNEL(ctx, lhs, rhs);

    if (lhs.isa<AShare>() && rhs.isa<AShare>()) {
      YASL_ENFORCE(lhs == rhs, "expect same, got lhs={}, rhs={}", lhs, rhs);
      ctx->setOutput(lhs);
    } else if (lhs.isa<AShare>() && rhs.isa<BShare>()) {
      ctx->setOutput(lhs);
    } else if (lhs.isa<BShare>() && rhs.isa<AShare>()) {
      ctx->setOutput(rhs);
    } else if (lhs.isa<BShare>() && rhs.isa<BShare>()) {
      ctx->setOutput(common_type_b(ctx->caller(), lhs, rhs));
    } else {
      YASL_THROW("should not be here, lhs={}, rhs={}", lhs, rhs);
    }
  }
};

class ABProtCastTypeS : public Kernel {
 public:
  static constexpr char kBindName[] = "cast_type_s";

  Kind kind() const override { return Kind::kDynamic; }

  void evaluate(EvalContext* ctx) const override {
    const auto& frm = ctx->getParam<ArrayRef>(0);
    const auto& to_type = ctx->getParam<Type>(1);

    SPU_TRACE_KERNEL(ctx, frm, to_type);

    if (frm.eltype().isa<AShare>() && to_type.isa<AShare>()) {
      YASL_ENFORCE(frm.eltype() == to_type,
                   "expect same, got frm={}, to_type={}", frm, to_type);
      // do nothing.
      ctx->setOutput(frm);
    } else if (frm.eltype().isa<AShare>() && to_type.isa<BShare>()) {
      ctx->setOutput(_A2B(frm));
    } else if (frm.eltype().isa<BShare>() && to_type.isa<AShare>()) {
      ctx->setOutput(_B2A(frm));
    } else if (frm.eltype().isa<BShare>() && to_type.isa<BShare>()) {
      ctx->setOutput(cast_type_b(ctx->caller(), frm, to_type));
    } else {
      YASL_THROW("should not be here, frm={}, to_type={}", frm, to_type);
    }
  }
};

class ABProtP2S : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_KERNEL(ctx, in);
    return _P2A(in);
  }
};

class ABProtS2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "s2p";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_KERNEL(ctx, in);
    if (_IsA(in)) {
      return _A2P(in);
    } else {
      YASL_ENFORCE(_IsB(in));
      return _B2P(in);
    }
  }
};

class ABProtNotS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_KERNEL(ctx, in);
    if (_LAZY_AB) {
      // TODO: Both A&B could handle not(invert).
      // if (in.eltype().isa<BShare>()) {
      //  return _NotB(in);
      //} else {
      //  YASL_ENFORCE(in.eltype().isa<AShare>());
      //  return _NotA(in);
      //}
      return _NotA(_2A(in));
    }
    return _NotA(in);
  }
};

class ABProtAddSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AddAP(_2A(lhs), rhs);
    }
    return _AddAP(lhs, rhs);
  }
};

class ABProtAddSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AddAA(_2A(lhs), _2A(rhs));
    }
    return _AddAA(lhs, rhs);
  }
};

class ABProtMulSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _MulAP(_2A(lhs), rhs);
    }
    return _MulAP(lhs, rhs);
  }
};

class ABProtMulSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_HasMulA1B() && _IsA(rhs) && _IsB(lhs) && _NBits(lhs) == 1) {
      return _MulA1B(rhs, lhs);
    }
    if (_HasMulA1B() && _IsA(lhs) && _IsB(rhs) && _NBits(rhs) == 1) {
      return _MulA1B(lhs, rhs);
    }

    if (_LAZY_AB) {
      return _MulAA(_2A(lhs), _2A(rhs));
    }
    return _MulAA(lhs, rhs);
  }
};

class ABProtMatMulSP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                size_t M, size_t N, size_t K) const override {
    SPU_TRACE_KERNEL(ctx, A, B);
    if (_LAZY_AB) {
      return _MatMulAP(_2A(A), B, M, N, K);
    }
    return _MatMulAP(A, B, M, N, K);
  }
};

class ABProtMatMulSS : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A, const ArrayRef& B,
                size_t M, size_t N, size_t K) const override {
    SPU_TRACE_KERNEL(ctx, A, B);
    if (_LAZY_AB) {
      return _MatMulAA(_2A(A), _2A(B), M, N, K);
    }
    return _MatMulAA(A, B, M, N, K);
  }
};

class ABProtAndSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AndBP(_2B(lhs), rhs);
    }
    return _B2A(_AndBP(_A2B(lhs), rhs));
  }
};

class ABProtAndSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AndBB(_2B(lhs), _2B(rhs));
    }
    return _B2A(_AndBB(_A2B(lhs), _A2B(rhs)));
  }
};

class ABProtXorSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_sp";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _XorBP(_2B(lhs), rhs);
    }
    return _B2A(_XorBP(_A2B(lhs), rhs));
  }
};

class ABProtXorSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_ss";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_KERNEL(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _XorBB(_2B(lhs), _2B(rhs));
    }
    return _B2A(_XorBB(_A2B(lhs), _A2B(rhs)));
  }
};

class ABProtEqzS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "eqz_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_KERNEL(ctx, in);
    //
    YASL_THROW("TODO");
  }
};

class ABProtLShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_KERNEL(ctx, in, bits);
    if (in.eltype().isa<AShare>()) {
      return _LShiftA(in, bits);
    } else if (in.eltype().isa<BShare>()) {
      if (_LAZY_AB) {
        return _LShiftB(in, bits);
      } else {
        return _B2A(_LShiftB(in, bits));
      }
    } else {
      YASL_THROW("Unsupported type {}", in.eltype());
    }
  }
};

class ABProtRShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_KERNEL(ctx, in, bits);
    if (_LAZY_AB) {
      return _RShiftB(_2B(in), bits);
    }
    return _B2A(_RShiftB(_A2B(in), bits));
  }
};

class ABProtARShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_KERNEL(ctx, in, bits);
    if (_LAZY_AB) {
      return _ARShiftB(_2B(in), bits);
    }
    return _B2A(_ARShiftB(_A2B(in), bits));
  }
};

class ABProtTruncPrS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "truncpr_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_KERNEL(ctx, in, bits);
    if (_LAZY_AB) {
      return _TruncPrA(_2A(in), bits);
    }
    return _TruncPrA(in, bits);
  }
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
                size_t end) const {
    SPU_TRACE_KERNEL(ctx, in, start, end);
    if (_LAZY_AB) {
      return _RitrevB(_2B(in), start, end);
    }
    return _B2A(_RitrevB(_A2B(in), start, end));
  }
};

class ABProtMsbS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_s";

  Kind kind() const override { return Kind::kDynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_KERNEL(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    if (ctx->caller()->hasKernel("msb_a")) {
      if (_LAZY_AB) {
        if (in.eltype().isa<BShare>()) {
          return _RShiftB(in, SizeOf(field) * 8 - 1);
        } else {
          // fast path, directly apply msb in AShare,
          // result a BShare.
          return _MsbA(in);
        }
      } else {
        // Do it in AShare domain, and convert back to
        // AShare.
        return _B2A(_MsbA(in));
      }
    } else {
      if (_LAZY_AB) {
        return _RShiftB(_2B(in), SizeOf(field) * 8 - 1);
      }
      return _B2A(_RShiftB(_A2B(in), SizeOf(field) * 8 - 1));
    }
  }
};

}  // namespace

Type common_type_b(Object* ctx, const Type& a, const Type& b) {
  return ctx->call<Type>("common_type_b", a, b);
}

ArrayRef cast_type_b(Object* ctx, const ArrayRef& a, const Type& to_type) {
  return ctx->call("cast_type_b", a, to_type);
}

ArrayRef zero_a(Object* ctx, FieldType field, size_t sz) {
  return ctx->call("zero_a", field, sz);
}

SPU_MPC_DEF_UNARY_OP(a2p)
SPU_MPC_DEF_UNARY_OP(p2a)
SPU_MPC_DEF_UNARY_OP(not_a)
SPU_MPC_DEF_UNARY_OP(msb_a)
SPU_MPC_DEF_BINARY_OP(add_ap)
SPU_MPC_DEF_BINARY_OP(add_aa)
SPU_MPC_DEF_BINARY_OP(mul_ap)
SPU_MPC_DEF_BINARY_OP(mul_aa)
SPU_MPC_DEF_BINARY_OP(mul_a1b)
SPU_MPC_DEF_UNARY_OP_WITH_SIZE(lshift_a)
SPU_MPC_DEF_UNARY_OP_WITH_SIZE(truncpr_a)
SPU_MPC_DEF_MMUL(mmul_ap)
SPU_MPC_DEF_MMUL(mmul_aa)

ArrayRef zero_b(Object* ctx, FieldType field, size_t sz) {
  return ctx->call("zero_b", field, sz);
}

SPU_MPC_DEF_UNARY_OP(b2p)
SPU_MPC_DEF_UNARY_OP(p2b)
SPU_MPC_DEF_UNARY_OP(a2b)
SPU_MPC_DEF_UNARY_OP(b2a)
SPU_MPC_DEF_BINARY_OP(and_bp)
SPU_MPC_DEF_BINARY_OP(and_bb)
SPU_MPC_DEF_BINARY_OP(xor_bp)
SPU_MPC_DEF_BINARY_OP(xor_bb)
SPU_MPC_DEF_UNARY_OP_WITH_SIZE(lshift_b)
SPU_MPC_DEF_UNARY_OP_WITH_SIZE(rshift_b)
SPU_MPC_DEF_UNARY_OP_WITH_SIZE(arshift_b)
SPU_MPC_DEF_UNARY_OP_WITH_2SIZE(bitrev_b)
SPU_MPC_DEF_BINARY_OP(add_bb)

void regABKernels(Object* obj) {
  obj->addState<ABProtState>();

  obj->regKernel<ABProtCommonTypeS>();
  obj->regKernel<ABProtCastTypeS>();
  obj->regKernel<ABProtP2S>();
  obj->regKernel<ABProtS2P>();
  obj->regKernel<ABProtNotS>();
  obj->regKernel<ABProtAddSP>();
  obj->regKernel<ABProtAddSS>();
  obj->regKernel<ABProtMulSP>();
  obj->regKernel<ABProtMulSS>();
  obj->regKernel<ABProtMatMulSP>();
  obj->regKernel<ABProtMatMulSS>();
  obj->regKernel<ABProtAndSP>();
  obj->regKernel<ABProtAndSS>();
  obj->regKernel<ABProtXorSP>();
  obj->regKernel<ABProtXorSS>();
  obj->regKernel<ABProtEqzS>();
  obj->regKernel<ABProtLShiftS>();
  obj->regKernel<ABProtRShiftS>();
  obj->regKernel<ABProtARShiftS>();
  obj->regKernel<ABProtTruncPrS>();
  obj->regKernel<ABProtBitrevS>();
  obj->regKernel<ABProtMsbS>();
}

#define COMMUTATIVE_DISPATCH(FnPP, FnBP, FnBB)     \
  if (_IsP(x) && _IsP(y)) {                        \
    return FnPP(ctx, x, y);                        \
  } else if (_IsB(x) && _IsP(y)) {                 \
    return FnBP(ctx, x, y);                        \
  } else if (_IsP(x) && _IsB(y)) {                 \
    /* commutative, swap args */                   \
    return FnBP(ctx, y, x);                        \
  } else if (_IsB(x) && _IsB(y)) {                 \
    return FnBB(ctx, y, x);                        \
  } else {                                         \
    YASL_THROW("unsupported op x={}, y={}", x, y); \
  }

CircuitBasicBlock<ArrayRef> makeABProtBasicBlock(Object* ctx) {
  CircuitBasicBlock<ArrayRef> cbb;
  cbb._xor = [=](ArrayRef const& x, ArrayRef const& y) -> ArrayRef {
    COMMUTATIVE_DISPATCH(xor_pp, xor_bp, xor_bb);
  };
  cbb._and = [=](ArrayRef const& x, ArrayRef const& y) -> ArrayRef {
    COMMUTATIVE_DISPATCH(and_pp, and_bp, and_bb);
  };
  cbb.lshift = [=](ArrayRef const& x, size_t bits) -> ArrayRef {
    if (_IsP(x)) {
      return lshift_p(ctx, x, bits);
    } else if (_IsB(x)) {
      return lshift_b(ctx, x, bits);
    }
    YASL_THROW("unsupported op x={}", x);
  };
  cbb.rshift = [=](ArrayRef const& x, size_t bits) -> ArrayRef {
    if (_IsP(x)) {
      return rshift_p(ctx, x, bits);
    } else if (_IsB(x)) {
      return rshift_b(ctx, x, bits);
    }
    YASL_THROW("unsupported op x={}", x);
  };
  cbb.init_like = [=](ArrayRef const& x, uint64_t hi, uint64_t lo) {
    // TODO: use single element + stride.
    const auto field = x.eltype().as<Ring2k>()->field();
    ArrayRef ret(makeType<Pub2kTy>(field), x.numel());

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using U = ring2k_t;
      U* ptr = &ret.at<U>(0);
      for (int64_t idx = 0; idx < x.numel(); idx++) {
        if constexpr (sizeof(U) * 8 == 128) {
          ptr[idx] = yasl::MakeUint128(hi, lo);
        } else {
          ptr[idx] = static_cast<U>(lo);
        }
      }
    });

    return ret;
  };
  cbb.set_nbits = [=](ArrayRef& x, size_t nbits) {
    YASL_ENFORCE(x.eltype().isa<BShare>());
    x.eltype().as<BShare>()->setNbits(nbits);
  };
  return cbb;
}

}  // namespace spu::mpc
