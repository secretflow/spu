// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/common/pv_gfmp.h"

#include <algorithm>
#include <mutex>

#include "magic_enum.hpp"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

inline bool isOwner(KernelEvalContext* ctx, const Type& type) {
  auto* comm = ctx->getState<Communicator>();
  return type.as<PrivGfmpTy>()->owner() ==
         static_cast<int64_t>(comm->getRank());
}

class P2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "p2v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override {
    auto* comm = ctx->getState<Communicator>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();
    const auto ty = makeType<PrivGfmpTy>(field, rank);
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
    const auto field = ctx->getState<Z2kState>()->getDefaultField();
    size_t owner = in.eltype().as<PrivGfmpTy>()->owner();
    auto* comm = ctx->getState<Communicator>();
    auto out_ty = makeType<PubGfmpTy>(field);
    NdArrayRef out = ring_zeros(field, in.shape());
    if (comm->getRank() == owner) {
      out = in;
    }
    out = comm->broadcast(out, owner, in.eltype(), in.shape(), "distribute");
    return out.as(out_ty);
  }
};

class MakeP : public Kernel {
 public:
  static constexpr const char* kBindName() { return "make_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->pushOutput(
        proc(ctx, ctx->getParam<uint128_t>(0), ctx->getParam<Shape>(1)));
  }

  static Value proc(KernelEvalContext* ctx, uint128_t init,
                    const Shape& shape) {
    const auto field = ctx->getState<Z2kState>()->getDefaultField();

    const auto eltype = makeType<PubGfmpTy>(field);
    auto buf = std::make_shared<yacl::Buffer>(1 * eltype.size());
    NdArrayRef arr(buf,                       // buffer
                   eltype,                    // eltype
                   shape,                     // shape
                   Strides(shape.size(), 0),  // strides
                   0);

    DISPATCH_ALL_FIELDS(field, [&]() {
      const auto* ty = eltype.as<GfmpTy>();
      const auto p = static_cast<ring2k_t>(ty->p());
      const auto mp_exp = ty->mp_exp();
      ring2k_t i = (init & p) + (init >> mp_exp);
      arr.at<ring2k_t>(Index(shape.size(), 0)) = i >= p ? i - p : i;
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
    const auto ty = makeType<PubGfmpTy>(field);
    auto r = prg_state->genPublWithMersennePrime(field, shape).as(ty);
    return r;
  }
};

class NegateP : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in) const override {
    NdArrayRef out(in.eltype(), in.shape());
    const auto* ty = in.eltype().as<GfmpTy>();
    const auto field = ty->field();
    const auto numel = in.numel();
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> _out(out);
      NdArrayView<ring2k_t> _in(in);
      pforeach(0, numel, [&](int64_t idx) { _out[idx] = add_inv(_in[idx]); });
    });
    return out;
  }
};

class NegateV : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_v"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    if (isOwner(ctx, in.eltype())) {
      NdArrayRef out(in.eltype(), in.shape());
      const auto* ty = in.eltype().as<GfmpTy>();
      const auto field = ty->field();
      const auto numel = in.numel();
      DISPATCH_ALL_FIELDS(field, [&]() {
        NdArrayView<ring2k_t> _out(out);
        NdArrayView<ring2k_t> _in(in);
        pforeach(0, numel, [&](int64_t idx) { _out[idx] = add_inv(_in[idx]); });
      });
      return out;
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
    const auto* ty = in.eltype().as<GfmpTy>();
    const auto field = ty->field();
    return ring_rshift(in,
                       {static_cast<int64_t>(GetMersennePrimeExp(field) - 1)})
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
      const auto* ty = in.eltype().as<GfmpTy>();
      const auto field = ty->field();
      return ring_rshift(in,
                         {static_cast<int64_t>(GetMersennePrimeExp(field) - 1)})
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
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());

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
    return gfmp_add_mod(lhs, rhs).as(lhs.eltype());
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
      return gfmp_add_mod(lhs, rhs).as(lhs.eltype());
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
      return gfmp_add_mod(lhs, rhs).as(lhs.eltype());
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
    return gfmp_mul_mod(lhs, rhs).as(lhs.eltype());
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
      return gfmp_mul_mod(lhs, rhs).as(lhs.eltype());
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
      return gfmp_mul_mod(lhs, rhs).as(lhs.eltype());
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
    return gfmp_mmul_mod(lhs, rhs).as(lhs.eltype());
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
    return gfmp_mmul_mod(lhs, rhs).as(lhs.eltype());
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
    return gfmp_mmul_mod(lhs, rhs).as(lhs.eltype());
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
    return gfmp_mod(ring_xor(lhs, rhs).as(lhs.eltype()));
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
      return gfmp_mod(ring_xor(lhs, rhs).as(lhs.eltype()));
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
      return gfmp_mod(ring_xor(lhs, rhs).as(lhs.eltype()));
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
    const auto* ty = in.eltype().as<GfmpTy>();
    const auto field = ty->field();
    bool is_splat = bits.size() == 1;
    auto max_bits = *std::max_element(bits.begin(), bits.end());
    SPU_ENFORCE_GT(GetMersennePrimeExp(field), static_cast<size_t>(max_bits));
    NdArrayRef out(in.eltype(), in.shape());
    return DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> _in(in);
      NdArrayView<ring2k_t> _out(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        auto shift_bits = is_splat ? bits[0] : bits[idx];
        _out[idx] = mul_mod(_in[idx], (static_cast<ring2k_t>(1) << shift_bits));
      });
      return out;
    });
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
      const auto* ty = in.eltype().as<GfmpTy>();
      const auto field = ty->field();
      bool is_splat = bits.size() == 1;
      auto max_bits = *std::max_element(bits.begin(), bits.end());
      SPU_ENFORCE_GT(GetMersennePrimeExp(field), static_cast<size_t>(max_bits));

      NdArrayRef out(in.eltype(), in.shape());
      return DISPATCH_ALL_FIELDS(field, [&]() {
        NdArrayView<ring2k_t> _in(in);
        NdArrayView<ring2k_t> _out(out);
        pforeach(0, in.numel(), [&](int64_t idx) {
          auto shift_bits = is_splat ? bits[0] : bits[idx];
          _out[idx] = mul_mod(_in[idx], static_cast<ring2k_t>(1) << shift_bits);
        });
        return out;
      });
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
    return gfmp_arshift_mod(in, bits);
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
      return gfmp_arshift_mod(in, bits);
    } else {
      return in;
    }
  }
};

class TruncP : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_p"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext*, const NdArrayRef& in,
                  const Sizes& bits) const override {
    // Todo: round?
    return gfmp_arshift_mod(in, bits);
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
      // Todo: round?
      return gfmp_arshift_mod(in, bits);
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
    SPU_ENFORCE(end <= GetMersennePrimeExp(field));

    return gfmp_mod(ring_bitrev(in, start, end).as(in.eltype()));
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
    SPU_ENFORCE(end <= GetMersennePrimeExp(field));

    if (isOwner(ctx, in.eltype())) {
      return gfmp_mod(ring_bitrev(in, start, end).as(in.eltype()));
    } else {
      return in;
    }
  }
};

}  // namespace

std::string PrivGfmpTy::toString() const {
  return fmt::format("{},{}", magic_enum::enum_name(field()), owner_);
}

void PrivGfmpTy::fromString(std::string_view str) {
  auto comma = str.find_first_of(',');
  auto field_str = str.substr(0, comma);
  auto owner_str = str.substr(comma + 1);
  auto field = magic_enum::enum_cast<FieldType>(field_str);
  SPU_ENFORCE(field.has_value(), "parse failed from={}", str);
  field_ = field.value();
  owner_ = std::stoll(std::string(owner_str));
}

void regPVGfmpTypes() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    TypeContext::getTypeContext()->addTypes<PubGfmpTy, PrivGfmpTy>();
  });
}

void regPVGfmpKernels(Object* obj) {
  obj->regKernel<V2P, P2V,                       //
                 MakeP, RandP,                   //
                 NegateV, NegateP,               //
                 EqualVVV, EqualVP, EqualPP,     //
                 AddVVV, AddVP, AddPP,           //
                 MulVVV, MulVP, MulPP,           //
                 MatMulVVV, MatMulVP, MatMulPP,  //
                 AndVVV, AndVP, AndPP,           //
                 XorVVV, XorVP, XorPP,           //
                 LShiftV, LShiftP,               //
                 RShiftV, RShiftP,               //
                 BitrevV, BitrevP,               //
                 ARShiftV, ARShiftP,             //
                 MsbV, MsbP,                     //
                 TruncV, TruncP                  //
                 >();
}

}  // namespace spu::mpc
