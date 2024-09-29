// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/api.h"

#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"

namespace spu::mpc {
namespace {

inline bool IsA(const MemRef& x) { return x.eltype().isa<ArithShare>(); }
inline bool IsB(const MemRef& x) { return x.eltype().isa<BoolShare>(); }
inline bool IsO(const MemRef& x) { return x.eltype().isa<OramShare>(); }
inline bool IsOP(const MemRef& x) { return x.eltype().isa<OramPubShare>(); }

inline bool IsPShr(const MemRef& x) { return x.eltype().isa<PermShare>(); }
[[maybe_unused]] inline bool IsP(const MemRef& x) {
  return x.eltype().isa<Public>();
}
[[maybe_unused]] inline bool IsV(const MemRef& x) {
  return x.eltype().isa<Private>();
}
inline size_t NBits(const MemRef& x) {
  return x.eltype().as<BaseRingType>()->valid_bits();
}
inline int64_t getOwner(const MemRef& x) {
  return x.eltype().as<Private>()->owner();
}
inline bool hasSameOwner(const MemRef& x, const MemRef& y) {
  return getOwner(x) == getOwner(y);
}

// NOLINTBEGIN(readability-identifier-naming)
MemRef _2b(SPUContext* ctx, const MemRef& x) {
  if (IsA(x)) {
    return a2b(ctx, x);
  } else {
    SPU_ENFORCE(IsB(x), "expect BShare, got {}", x.eltype());
    return x;
  }
}

MemRef _2a(SPUContext* ctx, const MemRef& x) {
  if (IsB(x)) {
    return b2a(ctx, x);
  } else {
    SPU_ENFORCE(IsA(x), "expect AShare, got {}", x.eltype());
    return x;
  }
}
// NOLINTEND(readability-identifier-naming)

// FIXME: move me to some where else.
#define IsS(X) false

// VSP dispatch rule.
// all,     commutative,  MPC aware
// f_ss,    f_ss,         f_ss
// f_sp,    f_sp,         f_sp
// f_sv,    f_sv,         f_sv(optional)
// f_ps,    _,            _
// f_pp,    f_pp,         _
// f_pv,    f_pv,         _
// f_vs,    _,            _
// f_vp,    _,            _
// f_vv,    f_vv,         f_vv or f_ss
template <typename FSS, typename FSV, typename FSP, typename FVV, typename FVP,
          typename FPP, typename... Args>
MemRef SvpBinaryDisp(SPUContext* ctx, const MemRef& x, const MemRef& y,
                     Args&&... args) {
  if (IsS(x)) {
    if (IsS(y)) {
      return FSS(ctx, x, y, std::forward<Args>(args)...);
    } else if (IsP(y)) {
      return FSP(ctx, x, y, std::forward<Args>(args)...);
    } else if (IsV(y)) {
      return FSV(ctx, x, y, std::forward<Args>(args)...);
    }
  } else if (IsV(x)) {
    if (IsS(y)) {
      return FSV(ctx, y, x, std::forward<Args>(args)...);
    } else if (IsP(y)) {
      return FVP(ctx, x, y, std::forward<Args>(args)...);
    } else if (IsV(y)) {
      return FVV(ctx, x, y, std::forward<Args>(args)...);
    }
  } else {
    SPU_ENFORCE(IsP(x));
    if (IsS(y)) {
      return FSP(ctx, y, x, std::forward<Args>(args)...);
    } else if (IsP(y)) {
      return FPP(ctx, x, y, std::forward<Args>(args)...);
    } else if (IsV(y)) {
      return FVP(ctx, y, x, std::forward<Args>(args)...);
    }
  }
}

template <typename FS, typename FV, typename FP, typename... Args>
MemRef SvpUnaryDisp(SPUContext* ctx, const MemRef& x, Args&&... args) {
  if (IsS(x)) {
    return FS(ctx, x, std::forward<Args>(args)...);
  } else if (IsV(x)) {
    return FV(ctx, x, std::forward<Args>(args)...);
  } else {
    SPU_ENFORCE(IsP(x));
    return FP(ctx, x, std::forward<Args>(args)...);
  }
}

}  // namespace

// TODO: Unify these macros.
#define FORCE_NAMED_DISPATCH(CTX, NAME, ...)      \
  {                                               \
    SPU_TRACE_MPC_LEAF(CTX, __VA_ARGS__);         \
    return dynDispatch((CTX), NAME, __VA_ARGS__); \
  }

#define FORCE_DISPATCH(CTX, ...) \
  FORCE_NAMED_DISPATCH(CTX, __func__, __VA_ARGS__)

#define TRY_NAMED_DISPATCH(CTX, FNAME, ...)        \
  if ((CTX)->hasKernel(FNAME)) {                   \
    SPU_TRACE_MPC_LEAF(CTX, __VA_ARGS__);          \
    return dynDispatch((CTX), FNAME, __VA_ARGS__); \
  }

#define TRY_DISPATCH(CTX, ...) TRY_NAMED_DISPATCH(CTX, __func__, __VA_ARGS__)

MemRef p2s(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);

  TRY_DISPATCH(ctx, x);

  return p2a(ctx, x);
}

MemRef p2v(SPUContext* ctx, const MemRef& x, size_t owner) {
  FORCE_DISPATCH(ctx, x, owner);
}

MemRef v2s(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);

  return v2a(ctx, x);
}

MemRef v2p(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef s2v(SPUContext* ctx, const MemRef& x, size_t owner) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x, owner);

  if (IsA(x)) {
    return a2v(ctx, x, owner);
  } else {
    SPU_ENFORCE(IsB(x));
    return b2v(ctx, x, owner);
  }
}

MemRef s2p(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);

  TRY_DISPATCH(ctx, x);

  if (IsA(x)) {
    return a2p(ctx, x);
  } else {
    SPU_ENFORCE(IsB(x), "invalid type {}", x.eltype());
    return b2p(ctx, x);
  }
}

MemRef import_s(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);

  TRY_DISPATCH(ctx, x);

  SPU_THROW("TODO: import_s not implemented");
}

MemRef export_s(SPUContext* ctx, const MemRef& x, const Type& t) {
  SPU_TRACE_MPC_DISP(ctx, x, t);

  TRY_DISPATCH(ctx, x, t);

  SPU_THROW("TODO: export_s not implemented");
}

MemRef ring_cast_p(SPUContext* ctx, const MemRef& in, PtType to_type) {
  SPU_TRACE_MPC_DISP(ctx, in, to_type);

  FORCE_DISPATCH(ctx, in, to_type);
}

MemRef ring_cast_s(SPUContext* ctx, const MemRef& in, PtType to_type) {
  SPU_TRACE_MPC_DISP(ctx, in, to_type);

  TRY_DISPATCH(ctx, in, to_type);

  // no-op
  return in;
}

Type common_type_s(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_MPC_DISP(ctx, a, b);

  // TRY_DISPATCH...
  if (ctx->hasKernel(__func__)) {
    SPU_TRACE_MPC_LEAF(ctx, a, b);
    return dynDispatch<Type>(ctx, __func__, a, b);
  }

  if (a.isa<ArithShare>() && b.isa<ArithShare>()) {
    return common_type_a(ctx, a, b);
  } else if (a.isa<ArithShare>() && b.isa<BoolShare>()) {
    return a;
  } else if (a.isa<BoolShare>() && b.isa<ArithShare>()) {
    return b;
  } else if (a.isa<BoolShare>() && b.isa<BoolShare>()) {
    return common_type_b(ctx, a, b);
  } else {
    SPU_THROW("should not be here, a={}, b={}", a, b);
  }
}

Type common_type_v(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_MPC_DISP(ctx, a, b);
  if (a == b) {
    return a;
  }
  return dynDispatch<Type>(ctx, __func__, a, b);
}

MemRef cast_type_s(SPUContext* ctx, const MemRef& frm, const Type& to_type) {
  SPU_TRACE_MPC_DISP(ctx, frm, to_type);

  TRY_DISPATCH(ctx, frm, to_type);

  if (IsA(frm) && to_type.isa<ArithShare>()) {
    return cast_type_a(ctx, frm, to_type);
  } else if (IsA(frm) && to_type.isa<BoolShare>()) {
    return a2b(ctx, frm);
  } else if (IsB(frm) && to_type.isa<ArithShare>()) {
    return b2a(ctx, frm);
  } else if (IsB(frm) && to_type.isa<BoolShare>()) {
    return cast_type_b(ctx, frm, to_type);
  } else {
    SPU_THROW("should not be here, frm={}, to_type={}", frm, to_type);
  }
}

MemRef make_p(SPUContext* ctx, uint128_t init, SemanticType type,
              const Shape& shape) {
  FORCE_DISPATCH(ctx, init, type, shape);
}

MemRef rand_p(SPUContext* ctx, SemanticType type, const Shape& shape) {
  FORCE_DISPATCH(ctx, type, shape);
}

MemRef rand_s(SPUContext* ctx, SemanticType type, const Shape& shape) {
  SPU_TRACE_MPC_DISP(ctx, shape);
  TRY_DISPATCH(ctx, type, shape);
  // always return random a share
  return rand_a(ctx, type, shape);
}

MemRef ring_cast_p(SPUContext* ctx, const MemRef& in, SemanticType to_type) {
  SPU_TRACE_MPC_DISP(ctx, in, to_type);
  FORCE_DISPATCH(ctx, in, to_type);
}

MemRef ring_cast_s(SPUContext* ctx, const MemRef& in, SemanticType to_type) {
  SPU_TRACE_MPC_DISP(ctx, in, to_type);
  FORCE_DISPATCH(ctx, in, to_type);
}

MemRef ring_cast_v(SPUContext* ctx, const MemRef& in, SemanticType to_type) {
  SPU_TRACE_MPC_DISP(ctx, in, to_type);
  FORCE_DISPATCH(ctx, in, to_type);
}

// only works for Z2k.
// Neg(x) = Not(x) + 1
// Not(x) = Neg(x) - 1
MemRef not_v(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  auto field = SizeOf(x.eltype().storage_type()) * 8;
  auto k1 = make_p(ctx, 1, GetSemanticType(field), x.shape());
  return add_vp(ctx, negate_v(ctx, x), negate_p(ctx, k1));
}

MemRef not_p(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  auto field = SizeOf(x.eltype().storage_type()) * 8;
  auto k1 = make_p(ctx, 1, GetSemanticType(field), x.shape());
  return add_pp(ctx, negate_p(ctx, x), negate_p(ctx, k1));
}

MemRef not_s(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  if (IsB(x)) {
    auto field = ctx->config().protocol().field();
    auto ones = make_p(ctx, -1, GetSemanticType(field), x.shape());
    return xor_bp(ctx, x, ones);
  } else {
    SPU_ENFORCE(x.eltype().isa<Secret>());
    auto field = SizeOf(x.eltype().storage_type()) * 8;
    auto k1 = make_p(ctx, 1, GetSemanticType(field), x.shape());
    return add_sp(ctx, negate_s(ctx, x), negate_p(ctx, k1));
  }
}

MemRef negate_s(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);
  return negate_a(ctx, _2a(ctx, x));
}

MemRef negate_v(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef negate_p(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

//////////////////////////////////////////////////////////////////////////////

MemRef msb_s(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);

  const auto back_type = x.eltype().storage_type();

  if (ctx->hasKernel("msb_a2b") && IsA(x)) {
    return msb_a2b(ctx, x);
  } else {
    auto shifted = rshift_b(ctx, _2b(ctx, x),
                            {static_cast<int64_t>(SizeOf(back_type) * 8 - 1)});
    shifted.eltype().as<BaseRingType>()->set_semantic_type(SE_1);
    return shifted;
  }
}

MemRef msb_v(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef msb_p(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

//////////////////////////////////////////////////////////////////////////////

MemRef equal_pp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<MemRef> equal_sp(SPUContext* ctx, const MemRef& x,
                             const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);

  if (IsA(x)) {
    TRY_NAMED_DISPATCH(ctx, "equal_ap", x, y);
  }
  if (IsB(x)) {
    TRY_NAMED_DISPATCH(ctx, "equal_bp", x, y);
  }

  return NotAvailable;
}

OptionalAPI<MemRef> equal_ss(SPUContext* ctx, const MemRef& x,
                             const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);

  // try fast path
  // TODO: use cost model instead of hand-coded priority.
  if (IsA(x) && IsA(y)) {
    TRY_NAMED_DISPATCH(ctx, "equal_aa", x, y);
  } else if (IsB(x) && IsB(y)) {
    TRY_NAMED_DISPATCH(ctx, "equal_bb", x, y);
  } else if ((IsA(x) && IsB(y)) || (IsB(x) && IsA(y))) {
    // mixed a & b, both OK, hardcode to a.
    if (ctx->hasKernel("equal_aa")) {
      FORCE_NAMED_DISPATCH(ctx, "equal_aa", _2a(ctx, x), _2a(ctx, y));
    }

    if (ctx->hasKernel("equal_bb")) {
      FORCE_NAMED_DISPATCH(ctx, "equal_bb", _2b(ctx, x), _2b(ctx, y));
    }
  }

  return NotAvailable;
}

//////////////////////////////////////////////////////////////////////////////

MemRef add_ss(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return add_aa(ctx, _2a(ctx, x), _2a(ctx, y));
}

MemRef add_sv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  // We can not use
  //   res = add_av(ctx, _2a(x), y)
  // since add_av is an optional API, so use `_2a` conversion to probe it is
  // not a good choice, i.e. if failed, a b2a maybe wasted.
  if (IsA(x)) {
    if (auto res = add_av(ctx, x, y)) {
      return res.value();
    }
  }
  return add_ss(ctx, x, v2s(ctx, y));
}

MemRef add_sp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return add_ap(ctx, _2a(ctx, x), y);
}

MemRef add_vv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "add_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "add_vvs", x, y);
    return add_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

MemRef add_vp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef add_pp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

static bool hasMulA1B(SPUContext* ctx) { return ctx->hasKernel("mul_a1b"); }

MemRef mul_ss(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);

  if (hasMulA1B(ctx) && IsA(y) && IsB(x) && NBits(x) == 1) {
    return mul_a1b(ctx, y, x);
  }
  if (hasMulA1B(ctx) && IsA(x) && IsB(y) && NBits(y) == 1) {
    return mul_a1b(ctx, x, y);
  }

  // NOTE(juhou): Multiplication of two bits
  if (IsB(x) && NBits(x) == 1 && IsB(y) && NBits(y) == 1) {
    return and_bb(ctx, x, y);
  }
  return mul_aa(ctx, _2a(ctx, x), _2a(ctx, y));
}

MemRef mul_sv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  if (IsA(x)) {
    if (auto res = mul_av(ctx, x, y)) {
      return res.value();
    }
  }
  return mul_ss(ctx, x, v2s(ctx, y));
}

MemRef mul_sp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return mul_ap(ctx, _2a(ctx, x), y);
}

MemRef mul_vv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "mul_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "mul_vvs", x, y);
    return mul_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

MemRef mul_vp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef mul_pp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef square_s(SPUContext* ctx, const MemRef& x) {
  if (IsA(x)) {
    TRY_NAMED_DISPATCH(ctx, "square_a", x);
  }
  return mul_ss(ctx, x, x);
}

MemRef square_v(SPUContext* ctx, const MemRef& x) { return mul_vv(ctx, x, x); }

MemRef square_p(SPUContext* ctx, const MemRef& x) { return mul_pp(ctx, x, x); }

//////////////////////////////////////////////////////////////////////////////

MemRef mmul_ss(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return mmul_aa(ctx, _2a(ctx, x), _2a(ctx, y));
}

MemRef mmul_sv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);

  if (ctx->hasKernel("mmul_av")) {
    // call a * v is available which is faster than calling a * a
    FORCE_NAMED_DISPATCH(ctx, "mmul_av", _2a(ctx, x), y);
  }

  // b * a will finally call a * a
  return mmul_ss(ctx, x, v2s(ctx, y));
}

MemRef mmul_sp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return mmul_ap(ctx, _2a(ctx, x), y);
}

MemRef mmul_vv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "mmul_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "mmul_vvs", x, y);
    return mmul_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

MemRef mmul_vp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef mmul_pp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

MemRef and_ss(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return and_bb(ctx, _2b(ctx, x), _2b(ctx, y));
}

MemRef and_sv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  if (IsA(x)) {
    if (auto res = and_bv(ctx, x, y)) {
      return res.value();
    }
  }
  return and_ss(ctx, x, v2s(ctx, y));
}

MemRef and_sp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return and_bp(ctx, _2b(ctx, x), y);
}

MemRef and_vv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "and_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "and_vvs", x, y);
    return and_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

MemRef and_vp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef and_pp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

MemRef xor_ss(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return xor_bb(ctx, _2b(ctx, x), _2b(ctx, y));
}

MemRef xor_sv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  if (IsA(x)) {
    if (auto res = xor_bv(ctx, x, y)) {
      return res.value();
    }
  }
  return xor_ss(ctx, x, v2s(ctx, y));
}

MemRef xor_sp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return xor_bp(ctx, _2b(ctx, x), y);
}

MemRef xor_vv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "xor_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "xor_vvs", x, y);
    return xor_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

MemRef xor_vp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef xor_pp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

MemRef lshift_s(SPUContext* ctx, const MemRef& x, const Sizes& bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  if (IsA(x)) {
    return lshift_a(ctx, x, bits);
  } else if (IsB(x)) {
    return lshift_b(ctx, x, bits);
  } else {
    SPU_THROW("Unsupported type {}", x.eltype());
  }
}

MemRef lshift_v(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

MemRef lshift_p(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

MemRef rshift_s(SPUContext* ctx, const MemRef& x, const Sizes& bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  return rshift_b(ctx, _2b(ctx, x), bits);
}

MemRef rshift_v(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

MemRef rshift_p(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

MemRef arshift_s(SPUContext* ctx, const MemRef& x, const Sizes& bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  return arshift_b(ctx, _2b(ctx, x), bits);
}

MemRef arshift_v(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

MemRef arshift_p(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

MemRef trunc_s(SPUContext* ctx, const MemRef& x, size_t bits, SignType sign) {
  SPU_TRACE_MPC_DISP(ctx, x, bits, sign);
  TRY_DISPATCH(ctx, x, bits, sign);
  return trunc_a(ctx, _2a(ctx, x), bits, sign);
}

MemRef trunc_v(SPUContext* ctx, const MemRef& x, size_t nbits, SignType sign) {
  FORCE_DISPATCH(ctx, x, nbits, sign);
}

MemRef trunc_p(SPUContext* ctx, const MemRef& x, size_t nbits, SignType sign) {
  // FIXME: trunc_p use shift kernel
  FORCE_DISPATCH(ctx, x, nbits, sign);
}

//////////////////////////////////////////////////////////////////////////////

MemRef bitrev_s(SPUContext* ctx, const MemRef& x, size_t start, size_t end) {
  SPU_TRACE_MPC_DISP(ctx, x, start, end);
  TRY_DISPATCH(ctx, x, start, end);
  return bitrev_b(ctx, _2b(ctx, x), start, end);
}

MemRef bitrev_v(SPUContext* ctx, const MemRef& x, size_t start, size_t end) {
  FORCE_DISPATCH(ctx, x, start, end);
}

MemRef bitrev_p(SPUContext* ctx, const MemRef& x, size_t start, size_t end) {
  FORCE_DISPATCH(ctx, x, start, end);
}

//////////////////////////////////////////////////////////////////////////////

OptionalAPI<MemRef> oram_onehot_ss(SPUContext* ctx, const MemRef& x,
                                   int64_t db_size) {
  SPU_TRACE_MPC_DISP(ctx, x, db_size);

  if (ctx->hasKernel("oram_onehot_aa")) {
    SPU_ENFORCE(IsA(x), "expect AShare, got {}", x.eltype());
    return dynDispatch(ctx, "oram_onehot_aa", x, db_size);
  }

  return NotAvailable;
}

OptionalAPI<MemRef> oram_onehot_sp(SPUContext* ctx, const MemRef& x,
                                   int64_t db_size) {
  SPU_TRACE_MPC_DISP(ctx, x, db_size);

  if (ctx->hasKernel("oram_onehot_ap")) {
    SPU_ENFORCE(IsA(x), "expect AShare, got {}", x.eltype());
    return dynDispatch(ctx, "oram_onehot_ap", x, db_size);
  }

  return NotAvailable;
}

MemRef oram_read_ss(SPUContext* ctx, const MemRef& x, const MemRef& y,
                    int64_t offset) {
  SPU_TRACE_MPC_DISP(ctx, x, offset);
  SPU_ENFORCE(IsO(x) && IsA(y), "expect OShare and AShare, got {} and {}",
              x.eltype(), y.eltype());

  return dynDispatch(ctx, "oram_read_aa", x, y, offset);
};

MemRef oram_read_sp(SPUContext* ctx, const MemRef& x, const MemRef& y,
                    int64_t offset) {
  SPU_TRACE_MPC_DISP(ctx, x, offset);
  SPU_ENFORCE(IsOP(x), "expect OPShare, got{}", x.eltype());

  return dynDispatch(ctx, "oram_read_ap", x, y, offset);
};

//////////////////////////////////////////////////////////////////////////////

OptionalAPI<MemRef> rand_perm_s(SPUContext* ctx, const Shape& shape) {
  SPU_TRACE_MPC_DISP(ctx, shape);
  TRY_NAMED_DISPATCH(ctx, "rand_perm_m", shape);
  return NotAvailable;
}

OptionalAPI<MemRef> perm_ss(SPUContext* ctx, const MemRef& x,
                            const MemRef& perm) {
  SPU_ENFORCE(IsPShr(perm), "perm should be a PShare");
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "perm_am", _2a(ctx, x), perm);
  return NotAvailable;
}

OptionalAPI<MemRef> perm_sp(SPUContext* ctx, const MemRef& x,
                            const MemRef& perm) {
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "perm_ap", _2a(ctx, x), perm);
  return NotAvailable;
}

spu::MemRef perm_pp(SPUContext* ctx, const MemRef& in, const MemRef& perm) {
  FORCE_DISPATCH(ctx, in, perm);
}

spu::MemRef perm_vv(SPUContext* ctx, const MemRef& in, const MemRef& perm) {
  SPU_ENFORCE(hasSameOwner(in, perm),
              "in and perm should belong to the same owner");
  FORCE_DISPATCH(ctx, in, perm);
}

OptionalAPI<MemRef> inv_perm_ss(SPUContext* ctx, const MemRef& x,
                                const MemRef& perm) {
  SPU_ENFORCE(IsPShr(perm), "perm should be a PShare");
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "inv_perm_am", _2a(ctx, x), perm);
  return NotAvailable;
}

OptionalAPI<MemRef> inv_perm_sp(SPUContext* ctx, const MemRef& x,
                                const MemRef& perm) {
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "inv_perm_ap", _2a(ctx, x), perm);
  return NotAvailable;
}

OptionalAPI<MemRef> inv_perm_sv(SPUContext* ctx, const MemRef& x,
                                const MemRef& perm) {
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "inv_perm_av", _2a(ctx, x), perm);
  return NotAvailable;
}

spu::MemRef inv_perm_pp(SPUContext* ctx, const MemRef& in, const MemRef& perm) {
  FORCE_DISPATCH(ctx, in, perm);
}

spu::MemRef inv_perm_vv(SPUContext* ctx, const MemRef& in, const MemRef& perm) {
  SPU_ENFORCE(hasSameOwner(in, perm),
              "in and perm should belong to the same owner");
  FORCE_DISPATCH(ctx, in, perm);
}

MemRef broadcast(SPUContext* ctx, const MemRef& in, const Shape& to_shape,
                 const Axes& in_dims) {
  SPU_TRACE_MPC_DISP(ctx, in, to_shape, in_dims);
  FORCE_DISPATCH(ctx, in, to_shape, in_dims);
}

// Resahpe a MemRef
MemRef reshape(SPUContext* ctx, const MemRef& in, const Shape& to_shape) {
  SPU_TRACE_MPC_DISP(ctx, in, to_shape);
  FORCE_DISPATCH(ctx, in, to_shape);
}

// Extract a slice from a MemRef
MemRef extract_slice(SPUContext* ctx, const MemRef& in, const Index& offsets,
                     const Shape& sizes, const Strides& strides) {
  SPU_TRACE_MPC_DISP(ctx, in, offsets, sizes, strides);
  FORCE_DISPATCH(ctx, in, offsets, sizes, strides);
}

// Update a MemRef at index with given MemRef
MemRef insert_slice(SPUContext* ctx, const MemRef& in, const MemRef& update,
                    const Index& offsets, const Strides& strides,
                    bool prefer_in_place) {
  SPU_TRACE_MPC_DISP(ctx, in, update, offsets, strides, prefer_in_place);
  FORCE_DISPATCH(ctx, in, update, offsets, strides, prefer_in_place);
}

// Transpose a MemRef
MemRef transpose(SPUContext* ctx, const MemRef& in, const Axes& permutation) {
  SPU_TRACE_MPC_DISP(ctx, in, permutation);
  FORCE_DISPATCH(ctx, in, permutation);
}

// Reverse a MemRef at dimensions
MemRef reverse(SPUContext* ctx, const MemRef& in, const Axes& dimensions) {
  SPU_TRACE_MPC_DISP(ctx, in, dimensions);
  FORCE_DISPATCH(ctx, in, dimensions);
}

// Fill a MemRef with input MemRef
MemRef fill(SPUContext* ctx, const MemRef& in, const Shape& to_shape) {
  SPU_TRACE_MPC_DISP(ctx, in, to_shape);
  FORCE_DISPATCH(ctx, in, to_shape);
}

// Pad a MemRef
MemRef pad(SPUContext* ctx, const MemRef& in, const MemRef& padding_MemRef,
           const Sizes& edge_padding_low, const Sizes& edge_padding_high) {
  SPU_TRACE_MPC_DISP(ctx, in, padding_MemRef, edge_padding_low,
                     edge_padding_high);
  FORCE_DISPATCH(ctx, in, padding_MemRef, edge_padding_low, edge_padding_high);
}

// Concate MemRefs at an axis
MemRef concatenate(SPUContext* ctx, const std::vector<MemRef>& MemRefs,
                   int64_t axis) {
  SPU_TRACE_MPC_DISP(ctx, MemRefs, axis);
  FORCE_DISPATCH(ctx, MemRefs, axis);
}

}  // namespace spu::mpc
