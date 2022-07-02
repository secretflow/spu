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
    YASL_ENFORCE(in.eltype().isa<AShare>());
    return in;
  }
}

#define _LAZY_AB ctx->caller()->getState<ABProtState>()->lazy_ab

#define _2A(x) _Lazy2A(ctx->caller(), x)
#define _2B(x) _Lazy2B(ctx->caller(), x)

#define _A2P(x) ctx->caller()->call("a2p", x)
#define _P2A(x) ctx->caller()->call("p2a", x)
#define _NotA(x) ctx->caller()->call("not_a", x)
#define _AddAP(lhs, rhs) ctx->caller()->call("add_ap", lhs, rhs)
#define _AddAA(lhs, rhs) ctx->caller()->call("add_aa", lhs, rhs)
#define _MulAP(lhs, rhs) ctx->caller()->call("mul_ap", lhs, rhs)
#define _MulAA(lhs, rhs) ctx->caller()->call("mul_aa", lhs, rhs)
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
}  // namespace

ArrayRef ABProtP2S::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_KERNEL(ctx, in);
  return _P2A(in);
}

ArrayRef ABProtS2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_KERNEL(ctx, in);
  if (_LAZY_AB) {
    return _A2P(_2A(in));
  }
  return _A2P(in);
}

ArrayRef ABProtNotS::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
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

ArrayRef ABProtAddSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _AddAP(_2A(lhs), rhs);
  }
  return _AddAP(lhs, rhs);
}

ArrayRef ABProtAddSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _AddAA(_2A(lhs), _2A(rhs));
  }
  return _AddAA(lhs, rhs);
}

ArrayRef ABProtMulSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _MulAP(_2A(lhs), rhs);
  }
  return _MulAP(lhs, rhs);
}

ArrayRef ABProtMulSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _MulAA(_2A(lhs), _2A(rhs));
  }
  return _MulAA(lhs, rhs);
}

ArrayRef ABProtMatMulSP::proc(KernelEvalContext* ctx, const ArrayRef& A,
                              const ArrayRef& B, size_t M, size_t N,
                              size_t K) const {
  SPU_TRACE_KERNEL(ctx, A, B);
  if (_LAZY_AB) {
    return _MatMulAP(_2A(A), B, M, N, K);
  }
  return _MatMulAP(A, B, M, N, K);
}

ArrayRef ABProtMatMulSS::proc(KernelEvalContext* ctx, const ArrayRef& A,
                              const ArrayRef& B, size_t M, size_t N,
                              size_t K) const {
  SPU_TRACE_KERNEL(ctx, A, B);
  if (_LAZY_AB) {
    return _MatMulAA(_2A(A), _2A(B), M, N, K);
  }
  return _MatMulAA(A, B, M, N, K);
}

ArrayRef ABProtAndSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _AndBP(_2B(lhs), rhs);
  }
  return _B2A(_AndBP(_A2B(lhs), rhs));
}

ArrayRef ABProtAndSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _AndBB(_2B(lhs), _2B(rhs));
  }
  return _B2A(_AndBB(_A2B(lhs), _A2B(rhs)));
}

ArrayRef ABProtXorSP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _XorBP(_2B(lhs), rhs);
  }
  return _B2A(_XorBP(_A2B(lhs), rhs));
}

ArrayRef ABProtXorSS::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                           const ArrayRef& rhs) const {
  SPU_TRACE_KERNEL(ctx, lhs, rhs);
  if (_LAZY_AB) {
    return _XorBB(_2B(lhs), _2B(rhs));
  }
  return _B2A(_XorBB(_A2B(lhs), _A2B(rhs)));
}

ArrayRef ABProtEqzS::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_KERNEL(ctx, in);
  //
  YASL_THROW("TODO");
}

ArrayRef ABProtLShiftS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                             size_t bits) const {
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

ArrayRef ABProtRShiftS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                             size_t bits) const {
  SPU_TRACE_KERNEL(ctx, in, bits);
  if (_LAZY_AB) {
    return _RShiftB(_2B(in), bits);
  }
  return _B2A(_RShiftB(_A2B(in), bits));
}

ArrayRef ABProtARShiftS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                              size_t bits) const {
  SPU_TRACE_KERNEL(ctx, in, bits);
  if (_LAZY_AB) {
    return _ARShiftB(_2B(in), bits);
  }
  return _B2A(_ARShiftB(_A2B(in), bits));
}

ArrayRef ABProtTruncPrS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                              size_t bits) const {
  SPU_TRACE_KERNEL(ctx, in, bits);
  if (_LAZY_AB) {
    return _TruncPrA(_2A(in), bits);
  }
  return _TruncPrA(in, bits);
}

ArrayRef ABProtBitrevS::proc(KernelEvalContext* ctx, const ArrayRef& in,
                             size_t start, size_t end) const {
  SPU_TRACE_KERNEL(ctx, in, start, end);
  if (_LAZY_AB) {
    return _RitrevB(_2B(in), start, end);
  }
  return _B2A(_RitrevB(_A2B(in), start, end));
}

ArrayRef ABProtMsbS::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_KERNEL(ctx, in);
  const auto field = in.eltype().as<Ring2k>()->field();
  if (ctx->caller()->hasKernel("MsbA")) {
    if (_LAZY_AB) {
      if (in.eltype().isa<BShare>()) {
        return _RShiftB(in, SizeOf(field) * 8 - 1);
      } else {
        // fast path, directly apply msb in AShare, result a BShare.
        return _MsbA(in);
      }
    } else {
      // Do it in AShare domain, and convert back to AShare.
      return _B2A(_MsbA(in));
    }
  } else {
    if (_LAZY_AB) {
      return _RShiftB(_2B(in), SizeOf(field) * 8 - 1);
    }
    return _B2A(_RShiftB(_A2B(in), SizeOf(field) * 8 - 1));
  }
}

}  // namespace spu::mpc
