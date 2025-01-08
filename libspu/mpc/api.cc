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

inline bool IsA(const Value& x) { return x.storage_type().isa<AShare>(); }
inline bool IsB(const Value& x) { return x.storage_type().isa<BShare>(); }
inline bool IsO(const Value& x) { return x.storage_type().isa<OShare>(); }
inline bool IsOP(const Value& x) { return x.storage_type().isa<OPShare>(); }

inline bool IsPShr(const Value& x) { return x.storage_type().isa<PShare>(); }
[[maybe_unused]] inline bool IsP(const Value& x) {
  return x.storage_type().isa<Public>();
}
[[maybe_unused]] inline bool IsV(const Value& x) {
  return x.storage_type().isa<Private>();
}
inline size_t NBits(const Value& x) {
  return x.storage_type().as<BShare>()->nbits();
}
inline int64_t getOwner(const Value& x) {
  return x.storage_type().as<Private>()->owner();
}
inline bool hasSameOwner(const Value& x, const Value& y) {
  return getOwner(x) == getOwner(y);
}

// NOLINTBEGIN(readability-identifier-naming)
Value _2b(SPUContext* ctx, const Value& x) {
  if (IsA(x)) {
    return a2b(ctx, x);
  } else {
    SPU_ENFORCE(IsB(x), "expect BShare, got {}", x.storage_type());
    return x;
  }
}

Value _2a(SPUContext* ctx, const Value& x) {
  if (IsB(x)) {
    return b2a(ctx, x);
  } else {
    SPU_ENFORCE(IsA(x), "expect AShare, got {}", x.storage_type());
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
Value SvpBinaryDisp(SPUContext* ctx, const Value& x, const Value& y,
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
Value SvpUnaryDisp(SPUContext* ctx, const Value& x, Args&&... args) {
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

Value p2s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);

  TRY_DISPATCH(ctx, x);

  if (x.dtype() == DT_I1) {
    return p2b(ctx, x);
  } else {
    return p2a(ctx, x);
  }
}

Value p2v(SPUContext* ctx, const Value& x, size_t owner) {
  FORCE_DISPATCH(ctx, x, owner);
}

Value v2s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);

  return v2a(ctx, x);
}

Value v2p(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value s2v(SPUContext* ctx, const Value& x, size_t owner) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x, owner);

  if (IsA(x)) {
    return a2v(ctx, x, owner);
  } else {
    SPU_ENFORCE(IsB(x));
    if (ctx->hasKernel("b2v")) {
      return b2v(ctx, x, owner);
    } else {
      return a2v(ctx, _2a(ctx, x), owner);
    }
  }
}

Value s2p(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);

  TRY_DISPATCH(ctx, x);

  if (IsA(x)) {
    return a2p(ctx, x);
  } else {
    SPU_ENFORCE(IsB(x), "invalid type {}", x.storage_type());
    return b2p(ctx, x);
  }
}

Value import_s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);

  TRY_DISPATCH(ctx, x);

  SPU_THROW("TODO: import_s not implemented");
}

Value export_s(SPUContext* ctx, const Value& x, const Type& t) {
  SPU_TRACE_MPC_DISP(ctx, x, t);

  TRY_DISPATCH(ctx, x, t);

  SPU_THROW("TODO: export_s not implemented");
}

Type common_type_s(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_MPC_DISP(ctx, a, b);

  // TRY_DISPATCH...
  if (ctx->hasKernel(__func__)) {
    SPU_TRACE_MPC_LEAF(ctx, a, b);
    return dynDispatch<Type>(ctx, __func__, a, b);
  }

  if (a.isa<AShare>() && b.isa<AShare>()) {
    SPU_ENFORCE(a == b, "expect same, got a={}, b={}", a, b);
    return a;
  } else if (a.isa<AShare>() && b.isa<BShare>()) {
    return a;
  } else if (a.isa<BShare>() && b.isa<AShare>()) {
    return b;
  } else if (a.isa<BShare>() && b.isa<BShare>()) {
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

Value cast_type_s(SPUContext* ctx, const Value& frm, const Type& to_type) {
  SPU_TRACE_MPC_DISP(ctx, frm, to_type);

  TRY_DISPATCH(ctx, frm, to_type);

  if (IsA(frm) && to_type.isa<AShare>()) {
    SPU_ENFORCE(frm.storage_type() == to_type,
                "expect same, got frm={}, to_type={}", frm, to_type);
    // do nothing.
    return frm;
  } else if (IsA(frm) && to_type.isa<BShare>()) {
    return a2b(ctx, frm);
  } else if (IsB(frm) && to_type.isa<AShare>()) {
    return b2a(ctx, frm);
  } else if (IsB(frm) && to_type.isa<BShare>()) {
    return cast_type_b(ctx, frm, to_type);
  } else {
    SPU_THROW("should not be here, frm={}, to_type={}", frm, to_type);
  }
}

Value make_p(SPUContext* ctx, uint128_t init, const Shape& shape) {
  FORCE_DISPATCH(ctx, init, shape);
}

Value rand_p(SPUContext* ctx, const Shape& shape) {
  FORCE_DISPATCH(ctx, shape);
}

Value rand_s(SPUContext* ctx, const Shape& shape, DataType dtype) {
  SPU_TRACE_MPC_DISP(ctx, shape);
  TRY_DISPATCH(ctx, shape);
  // can only get random bit share now.
  if (dtype == DT_I1) {
    return rand_b(ctx, shape);
  }
  // else, return random a share
  return rand_a(ctx, shape);
}

// only works for Z2k.
// Neg(x) = Not(x) + 1
// Not(x) = Neg(x) - 1
Value not_v(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  auto k1 = make_p(ctx, 1, x.shape());
  return add_vp(ctx, negate_v(ctx, x), negate_p(ctx, k1));
}

Value not_p(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  auto k1 = make_p(ctx, 1, x.shape());
  return add_pp(ctx, negate_p(ctx, x), negate_p(ctx, k1));
}

Value not_s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  if (x.storage_type().isa<BShare>()) {
    auto ones = make_p(ctx, -1, x.shape());
    return xor_bp(ctx, x, ones);
  } else {
    SPU_ENFORCE(x.storage_type().isa<Secret>());
    auto k1 = make_p(ctx, 1, x.shape());
    return add_sp(ctx, negate_s(ctx, x), negate_p(ctx, k1));
  }
}

Value negate_s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);
  return negate_a(ctx, _2a(ctx, x));
}

Value negate_v(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value negate_p(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

//////////////////////////////////////////////////////////////////////////////

Value msb_s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);

  // TODO: this is buggy.
  const auto field = ctx->getField();

  if (ctx->hasKernel("msb_a2b")) {
    if (IsB(x)) {
      return rshift_b(ctx, x, {static_cast<int64_t>(SizeOf(field) * 8 - 1)});
    } else {
      // fast path, directly apply msb x AShare, result a BShare.
      return msb_a2b(ctx, x);
    }
  } else if (ctx->hasKernel("msb_a") && IsA(x)) {
    return msb_a(ctx, x);
  } else {
    return rshift_b(ctx, _2b(ctx, x),
                    {static_cast<int64_t>(SizeOf(field) * 8 - 1)});
  }
}

Value msb_v(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value msb_p(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value relu(SPUContext* ctx, const Value& x) {FORCE_DISPATCH(ctx, x); }

//////////////////////////////////////////////////////////////////////////////

Value equal_pp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<Value> equal_sp(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);

  if (IsA(x)) {
    TRY_NAMED_DISPATCH(ctx, "equal_ap", x, y);
  }
  if (IsB(x)) {
    TRY_NAMED_DISPATCH(ctx, "equal_bp", x, y);
  }

  return NotAvailable;
}

OptionalAPI<Value> equal_ss(SPUContext* ctx, const Value& x, const Value& y) {
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

Value add_ss(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return add_aa(ctx, _2a(ctx, x), _2a(ctx, y));
}

Value add_sv(SPUContext* ctx, const Value& x, const Value& y) {
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

Value add_sp(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return add_ap(ctx, _2a(ctx, x), y);
}

Value add_vv(SPUContext* ctx, const Value& x, const Value& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "add_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "add_vvs", x, y);
    return add_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

Value add_vp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value add_pp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

static bool hasMulA1B(SPUContext* ctx) { return ctx->hasKernel("mul_a1b"); }

Value mul_ss(SPUContext* ctx, const Value& x, const Value& y) {
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

Value mul_sv(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  if (IsA(x)) {
    if (auto res = mul_av(ctx, x, y)) {
      return res.value();
    }
  }
  return mul_ss(ctx, x, v2s(ctx, y));
}

Value mul_sp(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return mul_ap(ctx, _2a(ctx, x), y);
}

Value mul_vv(SPUContext* ctx, const Value& x, const Value& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "mul_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "mul_vvs", x, y);
    return mul_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

Value mul_vp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value mul_pp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value square_s(SPUContext* ctx, const Value& x) {
  if (IsA(x)) {
    TRY_NAMED_DISPATCH(ctx, "square_a", x);
  }
  return mul_ss(ctx, x, x);
}

Value square_v(SPUContext* ctx, const Value& x) { return mul_vv(ctx, x, x); }

Value square_p(SPUContext* ctx, const Value& x) { return mul_pp(ctx, x, x); }

//////////////////////////////////////////////////////////////////////////////

Value mmul_ss(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return mmul_aa(ctx, _2a(ctx, x), _2a(ctx, y));
}

Value mmul_sv(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);

  if (ctx->hasKernel("mmul_av")) {
    // call a * v is available which is faster than calling a * a
    FORCE_NAMED_DISPATCH(ctx, "mmul_av", _2a(ctx, x), y);
  }

  // b * a will finally call a * a
  return mmul_ss(ctx, x, v2s(ctx, y));
}

Value mmul_sp(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return mmul_ap(ctx, _2a(ctx, x), y);
}

Value mmul_vv(SPUContext* ctx, const Value& x, const Value& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "mmul_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "mmul_vvs", x, y);
    return mmul_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

Value mmul_vp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value mmul_pp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

Value and_ss(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return and_bb(ctx, _2b(ctx, x), _2b(ctx, y));
}

Value and_sv(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  if (IsA(x)) {
    if (auto res = and_bv(ctx, x, y)) {
      return res.value();
    }
  }
  return and_ss(ctx, x, v2s(ctx, y));
}

Value and_sp(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return and_bp(ctx, _2b(ctx, x), y);
}

Value and_vv(SPUContext* ctx, const Value& x, const Value& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "and_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "and_vvs", x, y);
    return and_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

Value and_vp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value and_pp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

Value xor_ss(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return xor_bb(ctx, _2b(ctx, x), _2b(ctx, y));
}

Value xor_sv(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  if (IsA(x)) {
    if (auto res = xor_bv(ctx, x, y)) {
      return res.value();
    }
  }
  return xor_ss(ctx, x, v2s(ctx, y));
}

Value xor_sp(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);
  return xor_bp(ctx, _2b(ctx, x), y);
}

Value xor_vv(SPUContext* ctx, const Value& x, const Value& y) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "xor_vvv", x, y);
  } else {
    TRY_NAMED_DISPATCH(ctx, "xor_vvs", x, y);
    return xor_ss(ctx, v2s(ctx, x), v2s(ctx, y));
  }
}

Value xor_vp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value xor_pp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

//////////////////////////////////////////////////////////////////////////////

Value lshift_s(SPUContext* ctx, const Value& x, const Sizes& bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  if (IsA(x) && ctx->hasKernel("lshift_a")) {
    return lshift_a(ctx, x, bits);
  } else {
    return lshift_b(ctx, _2b(ctx, x), bits);
  }
}

Value lshift_v(SPUContext* ctx, const Value& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value lshift_p(SPUContext* ctx, const Value& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

Value rshift_s(SPUContext* ctx, const Value& x, const Sizes& bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  return rshift_b(ctx, _2b(ctx, x), bits);
}

Value rshift_v(SPUContext* ctx, const Value& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value rshift_p(SPUContext* ctx, const Value& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

Value arshift_s(SPUContext* ctx, const Value& x, const Sizes& bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  return arshift_b(ctx, _2b(ctx, x), bits);
}

Value arshift_v(SPUContext* ctx, const Value& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value arshift_p(SPUContext* ctx, const Value& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

Value trunc_s(SPUContext* ctx, const Value& x, size_t bits, SignType sign) {
  SPU_TRACE_MPC_DISP(ctx, x, bits, sign);
  TRY_DISPATCH(ctx, x, bits, sign);
  return trunc_a(ctx, _2a(ctx, x), bits, sign);
}

Value trunc_v(SPUContext* ctx, const Value& x, size_t nbits, SignType sign) {
  // FIXME: trunc_v use shift kernel
  const Sizes trunc_bits = {static_cast<int64_t>(nbits)};
  FORCE_DISPATCH(ctx, x, trunc_bits, sign);
}

Value trunc_p(SPUContext* ctx, const Value& x, size_t nbits, SignType sign) {
  // FIXME: trunc_p use shift kernel
  const Sizes trunc_bits = {static_cast<int64_t>(nbits)};
  FORCE_DISPATCH(ctx, x, trunc_bits, sign);
}

//////////////////////////////////////////////////////////////////////////////

Value bitrev_s(SPUContext* ctx, const Value& x, size_t start, size_t end) {
  SPU_TRACE_MPC_DISP(ctx, x, start, end);
  TRY_DISPATCH(ctx, x, start, end);
  return bitrev_b(ctx, _2b(ctx, x), start, end);
}

Value bitrev_v(SPUContext* ctx, const Value& x, size_t start, size_t end) {
  FORCE_DISPATCH(ctx, x, start, end);
}

Value bitrev_p(SPUContext* ctx, const Value& x, size_t start, size_t end) {
  FORCE_DISPATCH(ctx, x, start, end);
}

//////////////////////////////////////////////////////////////////////////////

OptionalAPI<Value> oram_onehot_ss(SPUContext* ctx, const Value& x,
                                  int64_t db_size) {
  SPU_TRACE_MPC_DISP(ctx, x, db_size);

  if (ctx->hasKernel("oram_onehot_aa")) {
    SPU_ENFORCE(IsA(x), "expect AShare, got {}", x.storage_type());
    return dynDispatch(ctx, "oram_onehot_aa", x, db_size);
  }

  return NotAvailable;
}

OptionalAPI<Value> oram_onehot_sp(SPUContext* ctx, const Value& x,
                                  int64_t db_size) {
  SPU_TRACE_MPC_DISP(ctx, x, db_size);

  if (ctx->hasKernel("oram_onehot_ap")) {
    SPU_ENFORCE(IsA(x), "expect AShare, got {}", x.storage_type());
    return dynDispatch(ctx, "oram_onehot_ap", x, db_size);
  }

  return NotAvailable;
}

Value oram_read_ss(SPUContext* ctx, const Value& x, const Value& y,
                   int64_t offset) {
  SPU_TRACE_MPC_DISP(ctx, x, offset);
  SPU_ENFORCE(IsO(x) && IsA(y), "expect OShare and AShare, got {} and {}",
              x.storage_type(), y.storage_type());

  return dynDispatch(ctx, "oram_read_aa", x, y, offset);
};

Value oram_read_sp(SPUContext* ctx, const Value& x, const Value& y,
                   int64_t offset) {
  SPU_TRACE_MPC_DISP(ctx, x, offset);
  SPU_ENFORCE(IsOP(x), "expect OPShare, got{}", x.storage_type());

  return dynDispatch(ctx, "oram_read_ap", x, y, offset);
};

//////////////////////////////////////////////////////////////////////////////

OptionalAPI<Value> rand_perm_s(SPUContext* ctx, const Shape& shape) {
  SPU_TRACE_MPC_DISP(ctx, shape);
  TRY_NAMED_DISPATCH(ctx, "rand_perm_m", shape);
  return NotAvailable;
}

OptionalAPI<Value> perm_ss(SPUContext* ctx, const Value& x, const Value& perm) {
  SPU_ENFORCE(IsPShr(perm), "perm should be a PShare");
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "perm_am", _2a(ctx, x), perm);
  return NotAvailable;
}

OptionalAPI<Value> perm_sp(SPUContext* ctx, const Value& x, const Value& perm) {
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "perm_ap", _2a(ctx, x), perm);
  return NotAvailable;
}

spu::Value perm_pp(SPUContext* ctx, const Value& in, const Value& perm) {
  FORCE_DISPATCH(ctx, in, perm);
}

spu::Value perm_vv(SPUContext* ctx, const Value& in, const Value& perm) {
  SPU_ENFORCE(hasSameOwner(in, perm),
              "in and perm should belong to the same owner");
  FORCE_DISPATCH(ctx, in, perm);
}

OptionalAPI<Value> inv_perm_ss(SPUContext* ctx, const Value& x,
                               const Value& perm) {
  SPU_ENFORCE(IsPShr(perm), "perm should be a PShare");
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "inv_perm_am", _2a(ctx, x), perm);
  return NotAvailable;
}

OptionalAPI<Value> inv_perm_sp(SPUContext* ctx, const Value& x,
                               const Value& perm) {
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "inv_perm_ap", _2a(ctx, x), perm);
  return NotAvailable;
}

OptionalAPI<Value> inv_perm_sv(SPUContext* ctx, const Value& x,
                               const Value& perm) {
  SPU_TRACE_MPC_DISP(ctx, x, perm);
  TRY_NAMED_DISPATCH(ctx, "inv_perm_av", _2a(ctx, x), perm);
  return NotAvailable;
}

spu::Value inv_perm_pp(SPUContext* ctx, const Value& in, const Value& perm) {
  FORCE_DISPATCH(ctx, in, perm);
}

spu::Value inv_perm_vv(SPUContext* ctx, const Value& in, const Value& perm) {
  SPU_ENFORCE(hasSameOwner(in, perm),
              "in and perm should belong to the same owner");
  FORCE_DISPATCH(ctx, in, perm);
}

Value broadcast(SPUContext* ctx, const Value& in, const Shape& to_shape,
                const Axes& in_dims) {
  SPU_TRACE_MPC_DISP(ctx, in, to_shape, in_dims);
  FORCE_DISPATCH(ctx, in, to_shape, in_dims);
}

// Resahpe a Value
Value reshape(SPUContext* ctx, const Value& in, const Shape& to_shape) {
  SPU_TRACE_MPC_DISP(ctx, in, to_shape);
  FORCE_DISPATCH(ctx, in, to_shape);
}

// Extract a slice from a Value
Value extract_slice(SPUContext* ctx, const Value& in,
                    const Index& start_indices, const Index& end_indices,
                    const Strides& strides) {
  SPU_TRACE_MPC_DISP(ctx, in, start_indices, end_indices, strides);
  FORCE_DISPATCH(ctx, in, start_indices, end_indices, strides);
}

// Update a Value at index with given value
Value update_slice(SPUContext* ctx, const Value& in, const Value& update,
                   const Index& start_indices) {
  SPU_TRACE_MPC_DISP(ctx, in, update, start_indices);
  FORCE_DISPATCH(ctx, in, update, start_indices);
}

// Transpose a Value
Value transpose(SPUContext* ctx, const Value& in, const Axes& permutation) {
  SPU_TRACE_MPC_DISP(ctx, in, permutation);
  FORCE_DISPATCH(ctx, in, permutation);
}

// Reverse a Value at dimensions
Value reverse(SPUContext* ctx, const Value& in, const Axes& dimensions) {
  SPU_TRACE_MPC_DISP(ctx, in, dimensions);
  FORCE_DISPATCH(ctx, in, dimensions);
}

// Fill a Value with input value
Value fill(SPUContext* ctx, const Value& in, const Shape& to_shape) {
  SPU_TRACE_MPC_DISP(ctx, in, to_shape);
  FORCE_DISPATCH(ctx, in, to_shape);
}

// Pad a Value
Value pad(SPUContext* ctx, const Value& in, const Value& padding_value,
          const Sizes& edge_padding_low, const Sizes& edge_padding_high,
          const Sizes& interior_padding) {
  SPU_TRACE_MPC_DISP(ctx, in, padding_value, edge_padding_low,
                     edge_padding_high, interior_padding);
  FORCE_DISPATCH(ctx, in, padding_value, edge_padding_low, edge_padding_high,
                 interior_padding);
}

// Concate Values at an axis
Value concatenate(SPUContext* ctx, const std::vector<Value>& values,
                  int64_t axis) {
  SPU_TRACE_MPC_DISP(ctx, values, axis);
  FORCE_DISPATCH(ctx, values, axis);
}

}  // namespace spu::mpc
