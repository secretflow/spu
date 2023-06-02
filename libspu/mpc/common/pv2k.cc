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

#include "libspu/mpc/common/pv2k.h"

#include <mutex>

#include "libspu/core/array_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

bool isOwner(KernelEvalContext* ctx, const ArrayRef& priv) {
  auto* comm = ctx->getState<Communicator>();
  return priv.eltype().as<Priv2kTy>()->owner() ==
         static_cast<int64_t>(comm->getRank());
}

class P2V : public RevealToKernel {
 public:
  static constexpr char kBindName[] = "p2v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t rank) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();
    const auto ty = makeType<Priv2kTy>(field, rank);
    if (comm->getRank() == rank) {
      return in.as(ty);
    } else {
      return makeConstantArrayRef(ty, in.numel());
    }
  }
};

class V2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "v2p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();
    size_t owner = in.eltype().as<Priv2kTy>()->owner();

    ArrayRef out(makeType<Pub2kTy>(field), in.numel());
    DISPATCH_ALL_FIELDS(field, "v2p", [&]() {
      std::vector<ring2k_t> priv(in.numel());
      pforeach(0, in.numel(), [&](int64_t idx) {  //
        priv[idx] = in.at<ring2k_t>(idx);
      });

      std::vector<ring2k_t> publ = comm->bcast<ring2k_t>(priv, owner, "v2p");
      pforeach(0, in.numel(),
               [&](int64_t idx) { out.at<ring2k_t>(idx) = publ[idx]; });
    });
    return out;
  }
};

class MakeP : public Kernel {
 public:
  static constexpr char kBindName[] = "make_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<uint128_t>(0), ctx->getParam<Shape>(1)));
  }

  static Value proc(KernelEvalContext* ctx, uint128_t init,
                    const Shape& shape) {
    const auto field = ctx->getState<Z2kState>()->getDefaultField();

    const auto eltype = makeType<Pub2kTy>(field);
    auto buf = std::make_shared<yacl::Buffer>(1 * eltype.size());
    NdArrayRef arr(buf,                                    // buffer
                   eltype,                                 // eltype
                   shape,                                  // shape
                   std::vector<int64_t>(shape.size(), 0),  // strides
                   0);

    DISPATCH_ALL_FIELDS(field, "pub2k.make_p", [&]() {
      arr.at<ring2k_t>(std::vector<int64_t>(shape.size(), 0)) =
          static_cast<ring2k_t>(init);
    });
    return Value(arr, DT_INVALID);
  }
};

class RandP : public RandKernel {
 public:
  static constexpr char kBindName[] = "rand_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, size_t size) const override {
    auto* prg_state = ctx->getState<PrgState>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();
    const auto ty = makeType<Pub2kTy>(field);
    return prg_state->genPubl(field, size).as(ty);
  }
};

class NotP : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_not(in).as(makeType<Pub2kTy>(field));
  }
};

class NotV : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    if (isOwner(ctx, in)) {
      return ring_not(in).as(in.eltype());
    } else {
      return in;
    }
  }
};

class MsbP : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return ring_rshift(in, in.elsize() * 8 - 1).as(in.eltype());
  }
};

class MsbV : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    if (isOwner(ctx, in)) {
      return ring_rshift(in, in.elsize() * 8 - 1).as(in.eltype());
    } else {
      return in;
    }
  }
};

class EqualPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_pp";

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());
    SPU_ENFORCE(x.eltype().isa<Pub2kTy>());

    return ring_equal(x, y).as(x.eltype());
  }
};

class EqualVVV : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_vvv";

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());

    if (isOwner(ctx, x)) {
      return ring_equal(x, y).as(x.eltype());
    } else {
      return x;
    }
  }
};

class EqualVP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_vp";

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());

    if (isOwner(ctx, x)) {
      return ring_equal(x, y).as(x.eltype());
    } else {
      return x;
    }
  }
};

class AddPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_pp";

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class AddVVV : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_vvv";

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());

    if (isOwner(ctx, lhs)) {
      return ring_add(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class AddVP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_vp";

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    if (isOwner(ctx, lhs)) {
      return ring_add(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class MulPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class MulVP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_vp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    if (isOwner(ctx, lhs)) {
      return ring_mul(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class MulVVV : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_vvv";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    if (isOwner(ctx, lhs)) {
      return ring_mul(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class MatMulPP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t m, size_t n,
                size_t k) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs, m, n, k).as(lhs.eltype());
  }
};

class MatMulVVV : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_vvv";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t m, size_t n,
                size_t k) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    if (isOwner(ctx, lhs)) {
      return ring_mmul(lhs, rhs, m, n, k).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class MatMulVP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_vp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t m, size_t n,
                size_t k) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    if (isOwner(ctx, lhs)) {
      return ring_mmul(lhs, rhs, m, n, k).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class AndPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class AndVVV : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_vvv";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    if (isOwner(ctx, lhs)) {
      return ring_and(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class AndVP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_vp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    if (isOwner(ctx, lhs)) {
      return ring_and(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class XorPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class XorVVV : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_vvv";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    if (isOwner(ctx, lhs)) {
      return ring_xor(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class XorVP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_vp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    if (isOwner(ctx, lhs)) {
      return ring_xor(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class LShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class LShiftV : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    if (isOwner(ctx, in)) {
      return ring_lshift(in, bits).as(in.eltype());
    } else {
      return in;
    }
  }
};

class RShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class RShiftV : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    if (isOwner(ctx, in)) {
      return ring_rshift(in, bits).as(in.eltype());
    } else {
      return in;
    }
  }
};

class ARShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class ARShiftV : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    if (isOwner(ctx, in)) {
      return ring_arshift(in, bits).as(in.eltype());
    } else {
      return in;
    }
  }
};

// TODO: move to utils and test it.
ArrayRef rounded_arshift(const ArrayRef& in, size_t bits) {
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

class TruncP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "trunc_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    return rounded_arshift(in, bits).as(in.eltype());
  }
};

class TruncV : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "trunc_v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    if (isOwner(ctx, in)) {
      return rounded_arshift(in, bits).as(in.eltype());
    } else {
      return in;
    }
  }
};

class BitrevP : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(start <= end);
    SPU_ENFORCE(end <= SizeOf(field) * 8);

    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class BitrevV : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_v";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(start <= end);
    SPU_ENFORCE(end <= SizeOf(field) * 8);

    if (isOwner(ctx, in)) {
      return ring_bitrev(in, start, end).as(in.eltype());
    } else {
      return in;
    }
  }
};

}  // namespace

void regPV2kTypes() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    TypeContext::getTypeContext()->addTypes<Pub2kTy, Priv2kTy>();
  });
}

void regPV2kKernels(Object* obj) {
  obj->regKernel<V2P>();
  obj->regKernel<P2V>();
  obj->regKernel<MakeP>();
  obj->regKernel<RandP>();
  obj->regKernel<NotV>();
  obj->regKernel<NotP>();
  obj->regKernel<EqualVVV>();
  obj->regKernel<EqualVP>();
  obj->regKernel<EqualPP>();
  obj->regKernel<AddVVV>();
  obj->regKernel<AddVP>();
  obj->regKernel<AddPP>();
  obj->regKernel<MulVVV>();
  obj->regKernel<MulVP>();
  obj->regKernel<MulPP>();
  obj->regKernel<MatMulVVV>();
  obj->regKernel<MatMulVP>();
  obj->regKernel<MatMulPP>();
  obj->regKernel<AndVVV>();
  obj->regKernel<AndVP>();
  obj->regKernel<AndPP>();
  obj->regKernel<XorVVV>();
  obj->regKernel<XorVP>();
  obj->regKernel<XorPP>();
  obj->regKernel<LShiftV>();
  obj->regKernel<LShiftP>();
  obj->regKernel<RShiftV>();
  obj->regKernel<RShiftP>();
  obj->regKernel<BitrevV>();
  obj->regKernel<BitrevP>();
  obj->regKernel<ARShiftV>();
  obj->regKernel<ARShiftP>();
  obj->regKernel<MsbV>();
  obj->regKernel<MsbP>();
  obj->regKernel<TruncV>();
  obj->regKernel<TruncP>();
}

}  // namespace spu::mpc
