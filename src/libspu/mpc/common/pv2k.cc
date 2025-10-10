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

#include "magic_enum.hpp"

#include "libspu/core/ndarray_ref.h"
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

inline int64_t getOwner(const NdArrayRef& x) {
  return x.eltype().as<Priv2kTy>()->owner();
}

class P2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "p2v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto field = in.eltype().as<Ring2k>()->field();
    const auto ty = makeType<Priv2kTy>(field, rank);
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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto field = in.eltype().as<Ring2k>()->field();
    size_t owner = in.eltype().as<Priv2kTy>()->owner();

    NdArrayRef out(makeType<Pub2kTy>(field), in.shape());

    auto numel = in.numel();

    DISPATCH_ALL_FIELDS(field, [&]() {
      std::vector<ring2k_t> priv(numel);
      NdArrayView<ring2k_t> _in(in);

      pforeach(0, numel, [&](int64_t idx) { priv[idx] = _in[idx]; });

      std::vector<ring2k_t> publ = comm->bcast<ring2k_t>(priv, owner, "v2p");

      NdArrayView<ring2k_t> _out(out);
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
                         ctx->getParam<Shape>(1), ctx->getParam<FieldType>(2)));
  }

  static Value proc(KernelEvalContext* ctx, uint128_t init, const Shape& shape,
                    FieldType field = FT_INVALID) {
    FieldType real_field = field;
    if (real_field == FT_INVALID) {
      real_field = ctx->getState<Z2kState>()->getDefaultField();
    }

    const auto eltype = makeType<Pub2kTy>(real_field);
    auto buf = std::make_shared<yacl::Buffer>(1 * eltype.size());
    NdArrayRef arr(buf,                       // buffer
                   eltype,                    // eltype
                   shape,                     // shape
                   Strides(shape.size(), 0),  // strides
                   0);

    DISPATCH_ALL_FIELDS(real_field, [&]() {
      arr.at<ring2k_t>(Index(shape.size(), 0)) = static_cast<ring2k_t>(init);
    });
    return Value(arr, DT_INVALID);
  }
};

class RandP : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override {
    auto* prg_state = ctx->getState<PrgState>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();
    const auto ty = makeType<Pub2kTy>(field);
    return prg_state->genPubl(field, shape).as(ty);
  }
};

class NegateP : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_neg(in).as(makeType<Pub2kTy>(field));
  }
};

class NegateV : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in) const override {
    return ring_rshift(in, {static_cast<int64_t>(in.elsize() * 8 - 1)})
        .as(in.eltype());
  }
};

class MsbV : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_rshift(in, {static_cast<int64_t>(in.elsize() * 8 - 1)})
          .as(in.eltype());
    } else {
      return in;
    }
  }
};

class EqualPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());
    SPU_ENFORCE(x.eltype().isa<Pub2kTy>());

    return ring_equal(x, y).as(x.eltype());
  }
};

class EqualVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());

    if (isOwner(ctx, x.eltype())) {
      return ring_equal(x, y).as(x.eltype());
    } else {
      return x;
    }
  }
};

class EqualVP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    if (isOwner(ctx, x.eltype())) {
      return ring_equal(x, y).as(x.eltype());
    } else {
      return x;
    }
  }
};

class AddPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class AddVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());

    if (isOwner(ctx, lhs.eltype())) {
      return ring_add(lhs, rhs).as(lhs.eltype());
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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_add(lhs, rhs).as(lhs.eltype());
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class MulVP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_mul(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class MulVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_mul(lhs, rhs).as(lhs.eltype());
    } else {
      return lhs;
    }
  }
};

class MatMulPP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs).as(lhs.eltype());
  }
};

class MatMulVVV : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    // For parties other than owner, also do a matmul to make result shape
    // correct.
    return ring_mmul(lhs, rhs).as(lhs.eltype());
  }
};

class MatMulVP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_vp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    // For parties other than owner, also do a matmul to make result shape
    // correct.
    return ring_mmul(lhs, rhs).as(lhs.eltype());
  }
};

class AndPP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class AndVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "and_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    if (isOwner(ctx, lhs.eltype())) {
      return ring_and(lhs, rhs).as(lhs.eltype());
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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_and(lhs, rhs).as(lhs.eltype());
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class XorVVV : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "xor_vvv"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_xor(lhs, rhs).as(lhs.eltype());
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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
    if (isOwner(ctx, lhs.eltype())) {
      return ring_xor(lhs, rhs).as(lhs.eltype());
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in,
                  const Sizes& bits) const override {
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class LShiftV : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_lshift(in, bits).as(in.eltype());
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in,
                  const Sizes& bits) const override {
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class RShiftV : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "rshift_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_rshift(in, bits).as(in.eltype());
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in,
                  const Sizes& bits) const override {
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class ARShiftV : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "arshift_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    if (isOwner(ctx, in.eltype())) {
      return ring_arshift(in, bits).as(in.eltype());
    } else {
      return in;
    }
  }
};

// TODO: move to utils and test it.
NdArrayRef rounded_arshift(const NdArrayRef& in, size_t bits) {
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

class TruncP : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in,
                  const Sizes& bits) const override {
    SPU_ENFORCE(bits.size() == 1, "truncation bits should be splat");
    return rounded_arshift(in, bits[0]).as(in.eltype());
  }
};

class TruncV : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    if (isOwner(ctx, in.eltype())) {
      SPU_ENFORCE(bits.size() == 1, "truncation bits should be splat");
      return rounded_arshift(in, bits[0]).as(in.eltype());
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in, size_t start,
                  size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(start <= end);
    SPU_ENFORCE(end <= SizeOf(field) * 8);

    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class BitrevV : public BitrevKernel {
 public:
  static constexpr const char* kBindName() { return "bitrev_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t start,
                  size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(start <= end);
    SPU_ENFORCE(end <= SizeOf(field) * 8);

    if (isOwner(ctx, in.eltype())) {
      return ring_bitrev(in, start, end).as(in.eltype());
    } else {
      return in;
    }
  }
};

namespace {
inline FieldType _get_field_from_n(size_t n) {
  if (n <= (static_cast<uint64_t>(1) << 8)) {
    return FieldType::FM8;
  } else if (n <= (static_cast<uint64_t>(1) << 16)) {
    return FieldType::FM16;
  } else if (n <= (static_cast<uint64_t>(1) << 32)) {
    return FieldType::FM32;
  } else {
    return FieldType::FM64;
  }
}
}  // namespace

class GenInvPermP : public GenInvPermKernel {
 public:
  static constexpr const char* kBindName() { return "gen_inv_perm_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in,
                  bool is_ascending) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    const auto perm_field = _get_field_from_n(in.numel());

    NdArrayRef out(makeType<Pub2kTy>(perm_field), in.shape());

    auto numel = in.numel();

    DISPATCH_ALL_FIELDS(perm_field, [&]() {
      using P = std::make_unsigned_t<ring2k_t>;
      DISPATCH_ALL_FIELDS(field, [&]() {
        using T = std::make_signed_t<ring2k_t>;
        std::vector<P> perm(numel);
        std::iota(perm.begin(), perm.end(), 0);
        // TODO: Add an iterator for NdArrayView
        NdArrayView<T> _in(in);
        NdArrayView<P> _out(out);
        auto cmp = [&_in, is_ascending](P a, P b) {
          return is_ascending ? _in[a] < _in[b] : _in[a] > _in[b];
        };
        std::stable_sort(perm.begin(), perm.end(), cmp);
        pforeach(0, numel, [&](int64_t idx) { _out[perm[idx]] = idx; });
      });
    });

    return out;
  }
};

class GenInvPermV : public GenInvPermKernel {
 public:
  static constexpr const char* kBindName() { return "gen_inv_perm_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  bool is_ascending) const override {
    if (isOwner(ctx, in.eltype())) {
      const auto perm_field = _get_field_from_n(in.numel());

      NdArrayRef out(
          makeType<Priv2kTy>(perm_field, in.eltype().as<Priv2kTy>()->owner()),
          in.shape());
      auto numel = in.numel();
      const auto field = in.eltype().as<Ring2k>()->field();

      DISPATCH_ALL_FIELDS(perm_field, [&]() {
        using P = std::make_unsigned_t<ring2k_t>;
        DISPATCH_ALL_FIELDS(field, [&]() {
          using T = std::make_signed_t<ring2k_t>;
          std::vector<P> perm(numel);
          std::iota(perm.begin(), perm.end(), 0);
          // TODO: Add an iterator for NdArrayView
          NdArrayView<T> _in(in);
          NdArrayView<P> _out(out);
          auto cmp = [&_in, is_ascending](P a, P b) {
            return is_ascending ? _in[a] < _in[b] : _in[a] > _in[b];
          };
          std::stable_sort(perm.begin(), perm.end(), cmp);
          pforeach(0, numel, [&](int64_t idx) { _out[perm[idx]] = idx; });
        });
      });

      return out;
    } else {
      const auto perm_field = _get_field_from_n(in.numel());
      return makeConstantArrayRef(
          makeType<Priv2kTy>(perm_field, in.eltype().as<Priv2kTy>()->owner()),
          in.shape());
    }
  }
};

class InvPermPP : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "inv_perm_pp"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    NdArrayRef z(x.eltype(), x.shape());
    const auto field = x.eltype().as<Ring2k>()->field();
    const auto perm_field = y.eltype().as<Ring2k>()->field();

    SPU_ENFORCE(perm_field == _get_field_from_n(x.numel()),
                "Invalid permutation field {}, should be {}",
                magic_enum::enum_name(perm_field),
                magic_enum::enum_name(_get_field_from_n(x.numel())));

    DISPATCH_ALL_FIELDS(perm_field, [&]() {
      using P = std::make_unsigned_t<ring2k_t>;
      DISPATCH_ALL_FIELDS(field, [&]() {
        using T = std::make_unsigned_t<ring2k_t>;
        NdArrayView<T> _x(x);
        NdArrayView<P> _y(y);
        NdArrayView<T> _z(z);
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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    const auto field = x.eltype().as<Ring2k>()->field();
    const auto perm_field = y.eltype().as<Ring2k>()->field();

    SPU_ENFORCE(perm_field == _get_field_from_n(x.numel()),
                "Invalid permutation field {}, should be {}",
                magic_enum::enum_name(perm_field),
                magic_enum::enum_name(_get_field_from_n(x.numel())));

    if (isOwner(ctx, x.eltype())) {
      NdArrayRef z(x.eltype(), x.shape());
      DISPATCH_ALL_FIELDS(perm_field, [&]() {
        using P = std::make_unsigned_t<ring2k_t>;
        DISPATCH_ALL_FIELDS(field, [&]() {
          using T = std::make_unsigned_t<ring2k_t>;
          NdArrayView<T> _x(x);
          NdArrayView<P> _y(y);
          NdArrayView<T> _z(z);
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

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    NdArrayRef z(x.eltype(), x.shape());
    const auto field = x.eltype().as<Ring2k>()->field();
    const auto perm_field = y.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(perm_field == _get_field_from_n(x.numel()),
                "Invalid permutation field {}, should be {}",
                magic_enum::enum_name(perm_field),
                magic_enum::enum_name(_get_field_from_n(x.numel())));

    DISPATCH_ALL_FIELDS(perm_field, [&]() {
      using P = std::make_unsigned_t<ring2k_t>;
      DISPATCH_ALL_FIELDS(field, [&]() {
        using T = std::make_unsigned_t<ring2k_t>;
        NdArrayView<T> _x(x);
        NdArrayView<P> _y(y);
        NdArrayView<T> _z(z);
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

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    const auto field = x.eltype().as<Ring2k>()->field();
    const auto perm_field = y.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(perm_field == _get_field_from_n(x.numel()),
                "Invalid permutation field {}, should be {}",
                magic_enum::enum_name(perm_field),
                magic_enum::enum_name(_get_field_from_n(x.numel())));

    if (isOwner(ctx, x.eltype())) {
      NdArrayRef z(x.eltype(), x.shape());

      DISPATCH_ALL_FIELDS(perm_field, [&]() {
        using P = std::make_unsigned_t<ring2k_t>;
        DISPATCH_ALL_FIELDS(field, [&]() {
          using T = std::make_unsigned_t<ring2k_t>;
          NdArrayView<T> _x(x);
          NdArrayView<P> _y(y);
          NdArrayView<T> _z(z);
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

  NdArrayRef proc(KernelEvalContext* ctx, absl::Span<NdArrayRef const> inputs,
                  bool is_ascending) const override {
    SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
    const auto field = inputs[0].eltype().as<Ring2k>()->field();

    SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                            [&field](const NdArrayRef& v) {
                              return v.eltype().as<Ring2k>()->field() == field;
                            }),
                "Inputs should belong to the same field");

    const auto numel = inputs[0].numel();
    const auto perm_field = _get_field_from_n(numel);
    NdArrayRef out(makeType<Pub2kTy>(perm_field), inputs[0].shape());

    DISPATCH_ALL_FIELDS(perm_field, [&]() {
      using P = std::make_unsigned_t<ring2k_t>;
      DISPATCH_ALL_FIELDS(field, [&]() {
        using T = std::make_unsigned_t<ring2k_t>;
        NdArrayView<P> _out(out);
        _out[0] = is_ascending ? (P)0 : static_cast<P>(numel - 1);
        for (int64_t i = 1; i < numel; ++i) {
          if (std::all_of(inputs.begin(), inputs.end(),
                          [i](const NdArrayRef& x) {
                            NdArrayView<T> _x(x);
                            return _x[i] == _x[i - 1];
                          })) {
            _out[i] = _out[i - 1];
          } else {
            _out[i] = is_ascending ? _out[i - 1] + 1 : _out[i - 1] - 1;
          }
        }
      });
    });

    return out;
  }
};

class MergeKeysV : public MergeKeysKernel {
 public:
  static constexpr const char* kBindName() { return "merge_keys_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, absl::Span<NdArrayRef const> inputs,
                  bool is_ascending) const override {
    SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
    SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                            [&inputs](const NdArrayRef& v) {
                              return (getOwner(v) == getOwner(inputs[0])) &&
                                     (v.eltype().as<Ring2k>()->field() ==
                                      inputs[0].eltype().as<Ring2k>()->field());
                            }),
                "Inputs should belong to the same owner");

    if (isOwner(ctx, inputs[0].eltype())) {
      const auto field = inputs[0].eltype().as<Ring2k>()->field();
      const auto numel = inputs[0].numel();
      const auto perm_field = _get_field_from_n(numel);
      NdArrayRef out(
          makeType<Priv2kTy>(perm_field,
                             inputs[0].eltype().as<Priv2kTy>()->owner()),
          inputs[0].shape());

      DISPATCH_ALL_FIELDS(perm_field, [&]() {
        using P = std::make_unsigned_t<ring2k_t>;
        DISPATCH_ALL_FIELDS(field, [&]() {
          using T = std::make_unsigned_t<ring2k_t>;
          NdArrayView<P> _out(out);
          _out[0] = is_ascending ? (P)0 : static_cast<P>(numel - 1);
          for (int64_t i = 1; i < numel; ++i) {
            if (std::all_of(inputs.begin(), inputs.end(),
                            [i](const NdArrayRef& x) {
                              NdArrayView<T> _x(x);
                              return _x[i] == _x[i - 1];
                            })) {
              _out[i] = _out[i - 1];
            } else {
              _out[i] = is_ascending ? _out[i - 1] + 1 : _out[i - 1] - 1;
            }
          }
        });
      });

      return out;
    } else {
      const auto perm_field = _get_field_from_n(inputs[0].numel());
      return makeConstantArrayRef(
          makeType<Priv2kTy>(perm_field,
                             inputs[0].eltype().as<Priv2kTy>()->owner()),
          inputs[0].shape());
    }
  }
};

class RingCastDownP : public RingCastDownKernel {
 public:
  static constexpr const char* kBindName() { return "ring_cast_down_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in,
                  FieldType to_field) const override {
    SPU_ENFORCE(in.eltype().isa<Pub2kTy>());
    const auto from_field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(SizeOf(from_field) >= SizeOf(to_field),
                "from_field={} to_field={}", magic_enum::enum_name(from_field),
                magic_enum::enum_name(to_field));

    if (from_field == to_field) {
      return in;
    }

    NdArrayRef out(makeType<Pub2kTy>(to_field), in.shape());

    DISPATCH_ALL_FIELDS(from_field, [&]() {
      using FromT = std::make_unsigned_t<ring2k_t>;
      DISPATCH_ALL_FIELDS(to_field, [&]() {
        using ToT = std::make_unsigned_t<ring2k_t>;
        NdArrayView<FromT> _in(in);
        NdArrayView<ToT> _out(out);
        pforeach(0, in.numel(),
                 [&](int64_t idx) { _out[idx] = static_cast<ToT>(_in[idx]); });
      });
    });

    return out;
  }
};

class RingCastDownV : public RingCastDownKernel {
 public:
  static constexpr const char* kBindName() { return "ring_cast_down_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }
  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  FieldType to_field) const override {
    SPU_ENFORCE(in.eltype().isa<Priv2kTy>());
    const auto from_field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(SizeOf(from_field) >= SizeOf(to_field),
                "from_field={} to_field={}", magic_enum::enum_name(from_field),
                magic_enum::enum_name(to_field));

    if (from_field == to_field) {
      return in;
    }

    if (isOwner(ctx, in.eltype())) {
      NdArrayRef out(
          makeType<Priv2kTy>(to_field, in.eltype().as<Priv2kTy>()->owner()),
          in.shape());

      DISPATCH_ALL_FIELDS(from_field, [&]() {
        using FromT = std::make_unsigned_t<ring2k_t>;
        DISPATCH_ALL_FIELDS(to_field, [&]() {
          using ToT = std::make_unsigned_t<ring2k_t>;
          NdArrayView<FromT> _in(in);
          NdArrayView<ToT> _out(out);
          pforeach(0, in.numel(), [&](int64_t idx) {
            _out[idx] = static_cast<ToT>(_in[idx]);
          });
        });
      });

      return out;
    } else {
      return makeConstantArrayRef(
          makeType<Priv2kTy>(to_field, in.eltype().as<Priv2kTy>()->owner()),
          in.shape());
    }
  }
};

}  // namespace

void Priv2kTy::fromString(std::string_view str) {
  auto comma = str.find_first_of(',');
  auto field_str = str.substr(0, comma);
  auto owner_str = str.substr(comma + 1);
  auto field = magic_enum::enum_cast<FieldType>(field_str);
  SPU_ENFORCE(field.has_value(), "parse failed from={}", str);
  field_ = field.value();
  owner_ = std::stoll(std::string(owner_str));
}

std::string Priv2kTy::toString() const {
  return fmt::format("{},{}", magic_enum::enum_name(field()), owner_);
}

void regPV2kTypes() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    TypeContext::getTypeContext()->addTypes<Pub2kTy, Priv2kTy>();
  });
}

void regPV2kKernels(Object* obj) {
  obj->regKernel<V2P, P2V,                                //
                 MakeP, RandP,                            //
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
                 RingCastDownP, RingCastDownV             //
                 >();
}

}  // namespace spu::mpc
