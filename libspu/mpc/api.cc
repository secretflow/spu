// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except x compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to x writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/api.h"

#include <future>

#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"

namespace spu::mpc {
namespace {

inline bool IsA(const Value& x) { return x.storage_type().isa<AShare>(); }
inline bool IsB(const Value& x) { return x.storage_type().isa<BShare>(); }
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

// FIXME: move me to some where elese.
#define IsS(X) false

// VSP dispath rule.
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
  if ((CTX)->hasKernel(__func__)) {                \
    SPU_TRACE_MPC_LEAF(CTX, __VA_ARGS__);          \
    return dynDispatch((CTX), FNAME, __VA_ARGS__); \
  }

#define TRY_DISPATCH(CTX, ...) TRY_NAMED_DISPATCH(CTX, __func__, __VA_ARGS__)

Value p2s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);

  TRY_DISPATCH(ctx, x);

  return p2a(ctx, x);
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
    return b2v(ctx, x, owner);
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
    return b;
  } else if (a.isa<BShare>() && b.isa<AShare>()) {
    return b;
  } else if (a.isa<BShare>() && b.isa<BShare>()) {
    return common_type_b(ctx, b, b);
  } else {
    SPU_THROW("should not be here, a={}, b={}", a, b);
  }
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

Value rand_s(SPUContext* ctx, const Shape& shape) {
  SPU_TRACE_MPC_DISP(ctx, shape);
  TRY_DISPATCH(ctx, shape);
  // always return random a share
  return rand_a(ctx, shape);
}

Value not_s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);
  // TODO: Both A&B could handle not(invert).
  // if (x.eltype().isa<BShare>()) {
  //  return not_b(ctx, x);
  //} else {
  //  SPU_ENFORCE(x.eltype().isa<AShare>());
  //  return not_a(ctx, x);
  //}
  return not_a(ctx, _2a(ctx, x));
}

Value not_v(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value not_p(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

//////////////////////////////////////////////////////////////////////////////

Value msb_s(SPUContext* ctx, const Value& x) {
  SPU_TRACE_MPC_DISP(ctx, x);
  TRY_DISPATCH(ctx, x);

  // TODO: this is buggy.
  const auto field = x.storage_type().as<Ring2k>()->field();

  if (ctx->hasKernel("msb_a2b")) {
    if (IsB(x)) {
      return rshift_b(ctx, x, SizeOf(field) * 8 - 1);
    } else {
      // fast path, directly apply msb x AShare, result a BShare.
      return msb_a2b(ctx, x);
    }
  } else {
    return rshift_b(ctx, _2b(ctx, x), SizeOf(field) * 8 - 1);
  }
}

Value msb_v(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value msb_p(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

//////////////////////////////////////////////////////////////////////////////

Value equal_pp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<Value> equal_sp(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);

  if (IsA(x) && ctx->hasKernel("equal_ap")) {
    return dynDispatch(ctx, "equal_ap", x, y);
  } else if (IsB(x) && ctx->hasKernel("equal_bp")) {
    return dynDispatch(ctx, "equal_bp", x, y);
  }

  return NotAvailable;
}

OptionalAPI<Value> equal_ss(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y);

  // try fast path
  // TODO: use cost model instead of hand-coded priority.
  if (IsA(x) && IsA(y) && ctx->hasKernel("equal_aa")) {
    return dynDispatch(ctx, "equal_aa", x, y);
  } else if (IsB(x) && IsB(y) && ctx->hasKernel("equal_bb")) {
    return dynDispatch(ctx, "equal_bb", x, y);
  } else if ((IsA(x) && IsB(y)) || (IsB(x) && IsA(y))) {
    // mixed a & b, both OK, hardcode to a.
    if (ctx->hasKernel("equal_aa")) {
      return dynDispatch(ctx, "equal_aa", _2a(ctx, x), _2a(ctx, y));
    }
    if (ctx->hasKernel("equal_bb")) {
      return dynDispatch(ctx, "equal_bb", _2b(ctx, x), _2b(ctx, y));
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

//////////////////////////////////////////////////////////////////////////////

Value mmul_ss(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  SPU_TRACE_MPC_DISP(ctx, x, y);
  TRY_DISPATCH(ctx, x, y, m, n, k);
  return mmul_aa(ctx, _2a(ctx, x), _2a(ctx, y), m, n, k);
}

Value mmul_sv(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  SPU_TRACE_MPC_DISP(ctx, x, y, m, n, k);
  TRY_DISPATCH(ctx, x, y, m, n, k);
  if (IsA(x)) {
    if (auto res = mmul_av(ctx, x, y, m, n, k)) {
      return res.value();
    }
  }
  return mmul_ss(ctx, x, v2s(ctx, y), m, n, k);
}

Value mmul_sp(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  SPU_TRACE_MPC_DISP(ctx, x, y, m, n, k);
  TRY_DISPATCH(ctx, x, y, m, n, k);
  return mmul_ap(ctx, _2a(ctx, x), y, m, n, k);
}

Value mmul_vv(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  if (hasSameOwner(x, y)) {
    FORCE_NAMED_DISPATCH(ctx, "mmul_vvv", x, y, m, n, k);
  } else {
    TRY_NAMED_DISPATCH(ctx, "mmul_vvs", x, y, m, n, k);
    return mmul_ss(ctx, v2s(ctx, x), v2s(ctx, y), m, n, k);
  }
}

Value mmul_vp(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  FORCE_DISPATCH(ctx, x, y, m, n, k);
}

Value mmul_pp(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  FORCE_DISPATCH(ctx, x, y, m, n, k);
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

Value lshift_s(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  if (IsA(x)) {
    return lshift_a(ctx, x, bits);
  } else if (IsB(x)) {
    return lshift_b(ctx, x, bits);
  } else {
    SPU_THROW("Unsupported type {}", x.storage_type());
  }
}

Value lshift_v(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value lshift_p(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

Value rshift_s(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  return rshift_b(ctx, _2b(ctx, x), bits);
}

Value rshift_v(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value rshift_p(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

Value arshift_s(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  return arshift_b(ctx, _2b(ctx, x), bits);
}

Value arshift_v(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value arshift_p(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

//////////////////////////////////////////////////////////////////////////////

Value trunc_s(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_MPC_DISP(ctx, x, bits);
  TRY_DISPATCH(ctx, x, bits);
  return trunc_a(ctx, _2a(ctx, x), bits);
}

Value trunc_v(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value trunc_p(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
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

}  // namespace spu::mpc
