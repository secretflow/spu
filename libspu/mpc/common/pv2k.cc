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

#include <algorithm>
#include <mutex>

#include "libspu/core/memref.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

inline bool isOwner(KernelEvalContext* ctx, const Type& type) {
  auto* comm = ctx->getState<Communicator>();
  return type.as<Priv2kTy>()->owner() == static_cast<int64_t>(comm->getRank());
}

inline int64_t getOwner(const MemRef& x) {
  return x.eltype().as<Priv2kTy>()->owner();
}

class P2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "p2v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              size_t rank) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto* in_type = in.eltype().as<Pub2kTy>();
    const auto ty = makeType<Priv2kTy>(in_type->semantic_type(), rank);
    if (comm->getRank() == rank) {
      return in.as(ty);
    } else {
      return makeConstantArrayRef(ty, in.shape());
    }
  }
};

class V2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override {
    auto* comm = ctx->getState<Communicator>();

    const auto* in_type = in.eltype().as<Priv2kTy>();
    size_t owner = in_type->owner();
    auto in_semantic_type = in_type->semantic_type();

    MemRef out(makeType<Pub2kTy>(in_semantic_type), in.shape());

    auto numel = in.numel();

    SPU_ENFORCE(in.eltype().storage_type() == out.eltype().storage_type(),
                "storage type mismatch, in = {}, out ={}",
                in.eltype().storage_type(), out.eltype().storage_type());

    DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
      std::vector<ScalarT> priv(numel);
      MemRefView<ScalarT> _in(in);

      pforeach(0, numel, [&](int64_t idx) { priv[idx] = _in[idx]; });

      std::vector<ScalarT> publ = comm->bcast<ScalarT>(priv, owner, "v2p");

      MemRefView<ScalarT> _out(out);
      pforeach(0, numel, [&](int64_t idx) { _out[idx] = publ[idx]; });
    });
    return out;
  }
};

class MakeP : public Kernel {
 public:
  static constexpr const char* kBindName() { return "make_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->pushOutput(proc(ctx, ctx->getParam<uint128_t>(0),
                         ctx->getParam<SemanticType>(1),
                         ctx->getParam<Shape>(2)));
  }

  static MemRef proc(KernelEvalContext* ctx, uint128_t init, SemanticType type,
                     const Shape& shape) {
    const auto eltype = makeType<Pub2kTy>(type);
    auto buf = std::make_shared<yacl::Buffer>(1 * eltype.size());
    MemRef arr(buf,                       // buffer
               eltype,                    // eltype
               shape,                     // shape
               Strides(shape.size(), 0),  // strides
               0);

    DISPATCH_ALL_STORAGE_TYPES(eltype.storage_type(), [&]() {
      MemRefView<ScalarT> _arr(arr);
      _arr[0] = static_cast<ScalarT>(init);
    });

    return arr;
  }
};

class RandP : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, SemanticType type,
              const Shape& shape) const override {
    auto* prg_state = ctx->getState<PrgState>();
    MemRef ret(makeType<Pub2kTy>(type), shape);

    prg_state->fillPubl(ret.data<std::byte>(), ret.numel() * ret.elsize());

    return ret;
  }
};

class NegateP : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in) const override {
    return ring_neg(in).as(in.eltype());
  }
};

class NegateV : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_neg(in).as(in.eltype());
    } else {
      return in;
    }
  }
};

class MsbP : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in) const override {
    MemRef res(makeType<Pub2kTy>(SE_1), in.shape());
    ring_msb(res, in);
    return res;
  }
};

class MsbV : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in) const override {
    const auto* in_type = in.eltype().as<Priv2kTy>();
    MemRef res(makeType<Priv2kTy>(SE_1, in_type->owner()), in.shape());
    auto owner = isOwner(ctx, in.eltype());

    if (owner) {
      ring_msb(res, in);
    } else {
      ring_zeros(res);
    }

    return res;
  }
};

class EqualPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& x,
              const MemRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());
    MemRef ret(makeType<Pub2kTy>(SE_1), x.shape());
    ring_equal(ret, x, y);
    return ret;
  }
};

class EqualVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& x,
              const MemRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());
    const auto* in_type = x.eltype().as<Priv2kTy>();
    auto owner = in_type->owner();

    MemRef ret(makeType<Priv2kTy>(SE_1, owner), x.shape());

    if (isOwner(ctx, x.eltype())) {
      ring_equal(ret, x, y);
    }

    return ret;
  }
};

class EqualVP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& x,
              const MemRef& y) const override {
    const auto* in_type = x.eltype().as<Priv2kTy>();
    auto owner = in_type->owner();

    MemRef ret(makeType<Priv2kTy>(SE_1, owner), x.shape());

    if (isOwner(ctx, x.eltype())) {
      ring_equal(ret, x, y);
    }

    return ret;
  }
};

class AddPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& lhs,
              const MemRef& rhs) const override {
    return ring_add(lhs, rhs);
  }
};

class AddVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());

    if (isOwner(ctx, lhs.eltype())) {
      return ring_add(lhs, rhs);
    } else {
      return lhs;
    }
  }
};

class AddVP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_add(lhs, rhs);
    } else {
      return lhs;
    }
  }
};

class MulPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& lhs,
              const MemRef& rhs) const override {
    auto ret = ring_mul(lhs, rhs);
    return ret.as(makeType<Pub2kTy>(ret.eltype().semantic_type()));
  }
};

class MulVP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      auto ret = ring_mul(lhs, rhs);
      return ret.as(
          makeType<Priv2kTy>(ret.eltype().semantic_type(), getOwner(lhs)));
    } else {
      auto lhs_st = lhs.eltype().semantic_type();
      auto rhs_st = rhs.eltype().semantic_type();
      auto ret_st = std::max(lhs_st, rhs_st);
      return makeConstantArrayRef(makeType<Priv2kTy>(ret_st, getOwner(lhs)),
                                  lhs.shape());
    }
  }
};

class MulVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      auto ret = ring_mul(lhs, rhs);
      return ret.as(
          makeType<Priv2kTy>(ret.eltype().semantic_type(), getOwner(lhs)));
    } else {
      auto lhs_st = lhs.eltype().semantic_type();
      auto rhs_st = rhs.eltype().semantic_type();
      auto ret_st = std::max(lhs_st, rhs_st);
      return makeConstantArrayRef(makeType<Priv2kTy>(ret_st, getOwner(lhs)),
                                  lhs.shape());
    }
  }
};

class MatMulPP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& lhs,
              const MemRef& rhs) const override {
    auto ret = ring_mmul(lhs, rhs);
    return ret.as(makeType<Pub2kTy>(ret.eltype().semantic_type()));
  }
};

class MatMulVVV : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    // For parties other than owner, also do a matmul to make result shape
    // correct.
    auto ret = ring_mmul(lhs, rhs);
    return ring_mmul(lhs, rhs).as(
        makeType<Priv2kTy>(ret.eltype().semantic_type(), getOwner(lhs)));
  }
};

class MatMulVP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    // For parties other than owner, also do a matmul to make result shape
    // correct.
    auto ret = ring_mmul(lhs, rhs);
    return ret.as(
        makeType<Priv2kTy>(ret.eltype().semantic_type(), getOwner(lhs)));
  }
};

class AndPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& lhs,
              const MemRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype(), "lhs = {}, rhs = {}",
                lhs.eltype(), rhs.eltype());
    return ring_and(lhs, rhs);
  }
};

class AndVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    if (isOwner(ctx, lhs.eltype())) {
      return ring_and(lhs, rhs);
    } else {
      return lhs;
    }
  }
};

class AndVP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_and(lhs, rhs);
    } else {
      return lhs;
    }
  }
};

class XorPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& lhs,
              const MemRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs);
  }
};

class XorVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_xor(lhs, rhs);
    } else {
      return lhs;
    }
  }
};

class XorVP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& lhs,
              const MemRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_xor(lhs, rhs);
    } else {
      return lhs;
    }
  }
};

class LShiftP : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in,
              const Sizes& bits) const override {
    return ring_lshift(in, bits);
  }
};

class LShiftV : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Sizes& bits) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_lshift(in, bits);
    } else {
      return in;
    }
  }
};

class RShiftP : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "rshift_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in,
              const Sizes& bits) const override {
    return ring_rshift(in, bits);
  }
};

class RShiftV : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "rshift_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Sizes& bits) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_rshift(in, bits);
    } else {
      return in;
    }
  }
};

class ARShiftP : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "arshift_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in,
              const Sizes& bits) const override {
    return ring_arshift(in, bits);
  }
};

class ARShiftV : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "arshift_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              const Sizes& bits) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_arshift(in, bits);
    } else {
      return in;
    }
  }
};

// TODO: move to utils and test it.
MemRef rounded_arshift(const MemRef& in, size_t bits) {
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

class TruncP : public TruncKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in, size_t bit) const override {
    return rounded_arshift(in, bit);
  }
};

class TruncV : public TruncKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              size_t bit) const override {
    if (isOwner(ctx, in.eltype())) {
      return rounded_arshift(in, bit);
    } else {
      return in;
    }
  }
};

class BitrevP : public BitrevKernel {
 public:
  static constexpr const char* kBindName() { return "bitrev_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in, size_t start,
              size_t end) const override {
    return ring_bitrev(in, start, end);
  }
};

class BitrevV : public BitrevKernel {
 public:
  static constexpr const char* kBindName() { return "bitrev_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in, size_t start,
              size_t end) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_bitrev(in, start, end);
    } else {
      return in;
    }
  }
};

class GenInvPermP : public GenInvPermKernel {
 public:
  static constexpr const char* kBindName() { return "gen_inv_perm_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in,
              bool is_ascending) const override {
    MemRef out(in.eltype(), in.shape());

    auto numel = in.numel();

    DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;
      std::vector<T> perm(numel);
      std::iota(perm.begin(), perm.end(), 0);
      // TODO: Add an iterator for MemRefView
      MemRefView<T> _in(in);
      MemRefView<T> _out(out);
      auto cmp = [&_in, is_ascending](int64_t a, int64_t b) {
        return is_ascending ? _in[a] < _in[b] : _in[a] > _in[b];
      };
      std::stable_sort(perm.begin(), perm.end(), cmp);
      for (int64_t idx = 0; idx < numel; ++idx) {
        _out[perm[idx]] = idx;
      }
    });
    return out;
  }
};

class GenInvPermV : public GenInvPermKernel {
 public:
  static constexpr const char* kBindName() { return "gen_inv_perm_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              bool is_ascending) const override {
    if (isOwner(ctx, in.eltype())) {
      MemRef out(in.eltype(), in.shape());
      auto numel = in.numel();

      DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
        using T = std::make_signed_t<ScalarT>;
        std::vector<T> perm(numel);
        std::iota(perm.begin(), perm.end(), 0);
        // TODO: Add an iterator for MemRefView
        MemRefView<T> _in(in);
        MemRefView<T> _out(out);
        auto cmp = [&_in, is_ascending](int64_t a, int64_t b) {
          return is_ascending ? _in[a] < _in[b] : _in[a] > _in[b];
        };
        std::stable_sort(perm.begin(), perm.end(), cmp);
        for (int64_t idx = 0; idx < numel; ++idx) {
          _out[perm[idx]] = idx;
        }
      });
      return out;
    } else {
      return in;
    }
  }
};

class InvPermPP : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "inv_perm_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& x,
              const MemRef& y) const override {
    MemRef z(x.eltype(), x.shape());
    DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;
      MemRefView<T> _x(x);
      MemRefView<T> _z(z);

      DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
        using T = std::make_signed_t<ScalarT>;
        MemRefView<T> _y(y);

        for (int64_t idx = 0; idx < x.numel(); ++idx) {
          _z[_y[idx]] = _x[idx];
        }
      });
    });
    return z;
  }
};

class InvPermVV : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "inv_perm_vv"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& x,
              const MemRef& y) const override {
    if (isOwner(ctx, x.eltype())) {
      MemRef z(x.eltype(), x.shape());
      DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
        using T = std::make_signed_t<ScalarT>;
        MemRefView<T> _x(x);
        MemRefView<T> _z(z);

        DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
          using T = std::make_signed_t<ScalarT>;
          MemRefView<T> _y(y);

          for (int64_t idx = 0; idx < x.numel(); ++idx) {
            _z[_y[idx]] = _x[idx];
          }
        });
      });
      return z;
    } else {
      return x;
    }
  }
};

class PermPP : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "perm_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& x,
              const MemRef& y) const override {
    MemRef z(x.eltype(), x.shape());
    DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;
      MemRefView<T> _x(x);
      MemRefView<T> _z(z);

      DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
        using T = std::make_signed_t<ScalarT>;
        MemRefView<T> _y(y);

        for (int64_t idx = 0; idx < x.numel(); ++idx) {
          _z[idx] = _x[_y[idx]];
        }
      });
    });
    return z;
  }
};

class PermVV : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "perm_vv"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& x,
              const MemRef& y) const override {
    if (isOwner(ctx, x.eltype())) {
      MemRef z(x.eltype(), x.shape());
      DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
        using T = std::make_signed_t<ScalarT>;
        MemRefView<T> _x(x);
        MemRefView<T> _z(z);

        DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
          using T = std::make_signed_t<ScalarT>;
          MemRefView<T> _y(y);

          for (int64_t idx = 0; idx < x.numel(); ++idx) {
            _z[idx] = _x[_y[idx]];
          }
        });
      });
      return z;
    } else {
      return x;
    }
  }
};

class MergeKeysP : public MergeKeysKernel {
 public:
  static constexpr const char* kBindName() { return "merge_keys_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, absl::Span<MemRef const> inputs,
              bool is_ascending) const override {
    SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
    MemRef out(inputs[0].eltype(), inputs[0].shape());
    const auto numel = inputs[0].numel();
    DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;
      MemRefView<T> _out(out);
      _out[0] = 0;
      for (int64_t i = 1; i < numel; ++i) {
        if (std::all_of(inputs.begin(), inputs.end(), [i](const MemRef& x) {
              MemRefView<T> _x(x);
              return _x[i] == _x[i - 1];
            })) {
          _out[i] = _out[i - 1];
        } else {
          _out[i] = is_ascending ? _out[i - 1] + 1 : _out[i - 1] - 1;
        }
      }
    });
    return out;
  }
};

class MergeKeysV : public MergeKeysKernel {
 public:
  static constexpr const char* kBindName() { return "merge_keys_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, absl::Span<MemRef const> inputs,
              bool is_ascending) const override {
    SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
    SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                            [&inputs](const MemRef& v) {
                              return getOwner(v) == getOwner(inputs[0]);
                            }),
                "Inputs should belong to the same owner");

    if (isOwner(ctx, inputs[0].eltype())) {
      MemRef out(inputs[0].eltype(), inputs[0].shape());
      const auto numel = inputs[0].numel();
      DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
        using T = std::make_signed_t<ScalarT>;
        MemRefView<T> _out(out);
        _out[0] = 0;
        for (int64_t i = 1; i < numel; ++i) {
          if (std::all_of(inputs.begin(), inputs.end(), [i](const MemRef& x) {
                MemRefView<T> _x(x);
                return _x[i] == _x[i - 1];
              })) {
            _out[i] = _out[i - 1];
          } else {
            _out[i] = is_ascending ? _out[i - 1] + 1 : _out[i - 1] - 1;
          }
        }
      });
      return out;
    } else {
      return makeConstantArrayRef(inputs[0].eltype(), inputs[0].shape());
    }
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

class RingCastP : public RingCastKernel {
 public:
  static constexpr const char* kBindName() { return "ring_cast_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext*, const MemRef& in,
              SemanticType to_type) const override {
    MemRef out(makeType<Pub2kTy>(to_type), in.shape());

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

class RingCastV : public RingCastKernel {
 public:
  static constexpr const char* kBindName() { return "ring_cast_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  MemRef proc(KernelEvalContext* ctx, const MemRef& in,
              SemanticType to_type) const override {
    auto owner = in.eltype().as<Priv2kTy>()->owner();

    auto result_type = makeType<Priv2kTy>(to_type, owner);

    if (isOwner(ctx, result_type)) {
      MemRef out(result_type, in.shape());

      bool inUnsigned = isUnsigned(in.eltype().semantic_type());
      bool outUnsigned = isUnsigned(out.eltype().semantic_type());

      DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
        using InT = ScalarT;
        DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
          using OutT = ScalarT;
          if (inUnsigned && outUnsigned) {
            copyRing<InT, OutT>(out, in);
          } else if (!inUnsigned && !outUnsigned) {
            copyRing<std::make_signed_t<InT>, std::make_signed_t<OutT>>(out,
                                                                        in);
          } else if (!inUnsigned) {
            copyRing<std::make_signed_t<InT>, OutT>(out, in);
          } else {
            copyRing<InT, std::make_signed_t<OutT>>(out, in);
          }
        });
      });

      return out;

    } else {
      return makeConstantArrayRef(result_type, in.shape());
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
  obj->regKernel<V2P, P2V,                                //
                 RandP, MakeP,                            //
                 NegateV, NegateP,                        //
                 EqualVVV, EqualVP, EqualPP,              //
                 AddVVV, AddVP, AddPP,                    //
                 MulVVV, MulVP, MulPP,                    //
                 MatMulVVV, MatMulVP, MatMulPP,           //
                 AndVVV, AndVP, AndPP,                    //
                 XorVVV, XorVP, XorPP,                    //
                 LShiftV, LShiftP,                        //
                 RShiftV, RShiftP,                        //
                 BitrevV, BitrevP,                        //
                 ARShiftV, ARShiftP,                      //
                 MsbV, MsbP,                              //
                 TruncV, TruncP,                          //
                 GenInvPermV, GenInvPermP,                //
                 InvPermPP, InvPermVV,                    //
                 PermPP, PermVV, MergeKeysP, MergeKeysV,  //
                 RingCastP, RingCastV>();
}

}  // namespace spu::mpc
