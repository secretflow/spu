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

#include "libspu/mpc/common/ab_api.h"

#include <future>

#include "libspu/core/parallel_utils.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/common/pub2k.h"

namespace spu::mpc {
namespace {

// TODO(jint) may be we should move tiling to a `tiling` layer or dialect.
ArrayRef block_par_unary(KernelEvalContext* ctx, std::string_view fn_name,
                         const ArrayRef& in) {
  const int64_t kBlockSize = kMinTaskSize;
  if (!ctx->caller()->hasLowCostFork() || in.numel() <= kBlockSize) {
    return ctx->caller()->call(fn_name, in);
  }

  std::string kBindName(fn_name);
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* obj = ctx->caller();
  std::vector<std::unique_ptr<Object>> sub_objs;

  const int64_t numBlocks =
      in.numel() / kBlockSize + ((in.numel() % kBlockSize) != 0 ? 1 : 0);

  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    sub_objs.push_back(obj->fork());
  }

  std::vector<std::future<ArrayRef>> futures;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    futures.push_back(std::async(
        [&](int64_t index) {
          int64_t begin = index * kBlockSize;
          int64_t end = std::min(begin + kBlockSize, in.numel());

          return sub_objs[index]->call(fn_name, in.slice(begin, end));
        },
        blk_idx));
  }

  std::vector<ArrayRef> out_slices;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    out_slices.push_back(futures[blk_idx].get());
  }

  // Assume out.numel = in.numel
  ArrayRef out(out_slices[0].eltype(), in.numel());
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    int64_t begin = blk_idx * kBlockSize;
    int64_t end = std::min(begin + kBlockSize, in.numel());
    std::memcpy(&out.at(begin), &out_slices[blk_idx].at(0),
                (end - begin) * out.elsize());
  }

  return out;
}

ArrayRef block_par_unary_with_size(KernelEvalContext* ctx,
                                   std::string_view fn_name, const ArrayRef& in,
                                   size_t bits) {
  const int64_t kBlockSize = kMinTaskSize;
  if (!ctx->caller()->hasLowCostFork() || in.numel() <= kBlockSize) {
    return ctx->caller()->call(fn_name, in, bits);
  }

  std::string kBindName(fn_name);
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* obj = ctx->caller();
  std::vector<std::unique_ptr<Object>> sub_objs;

  const int64_t numBlocks =
      in.numel() / kBlockSize + ((in.numel() % kBlockSize) != 0 ? 1 : 0);

  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    sub_objs.push_back(obj->fork());
  }

  std::vector<std::future<ArrayRef>> futures;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    futures.push_back(std::async(
        [&](int64_t index) {
          int64_t begin = index * kBlockSize;
          int64_t end = std::min(begin + kBlockSize, in.numel());
          return sub_objs[index]->call(fn_name, in.slice(begin, end), bits);
        },
        blk_idx));
  }

  std::vector<ArrayRef> out_slices;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    out_slices.push_back(futures[blk_idx].get());
  }

  // Assume out.numel = in.numel
  ArrayRef out(out_slices[0].eltype(), in.numel());
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    int64_t begin = blk_idx * kBlockSize;
    int64_t end = std::min(begin + kBlockSize, in.numel());
    std::memcpy(&out.at(begin), &out_slices[blk_idx].at(0),
                (end - begin) * out.elsize());
  }

  return out;
}

ArrayRef block_par_binary(KernelEvalContext* ctx, std::string_view fn_name,
                          const ArrayRef& lhs, const ArrayRef& rhs) {
  const int64_t kBlockSize = kMinTaskSize;
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  if (!ctx->caller()->hasLowCostFork() || lhs.numel() <= kBlockSize) {
    return ctx->caller()->call(fn_name, lhs, rhs);
  }

  const int64_t numel = lhs.numel();

  std::string kBindName(fn_name);
  SPU_TRACE_MPC_LEAF(ctx, lhs);

  auto* obj = ctx->caller();
  std::vector<std::unique_ptr<Object>> sub_objs;

  const int64_t numBlocks =
      numel / kBlockSize + ((numel % kBlockSize) != 0 ? 1 : 0);

  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    sub_objs.push_back(obj->fork());
  }

  std::vector<std::future<ArrayRef>> futures;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    futures.push_back(std::async(
        [&](int64_t index) {
          int64_t begin = index * kBlockSize;
          int64_t end = std::min(begin + kBlockSize, numel);

          return sub_objs[index]->call(fn_name, lhs.slice(begin, end),
                                       rhs.slice(begin, end));
        },
        blk_idx));
  }

  std::vector<ArrayRef> out_slices;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    out_slices.push_back(futures[blk_idx].get());
  }

  // Assume out.numel = numel
  ArrayRef out(out_slices[0].eltype(), numel);
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    int64_t begin = blk_idx * kBlockSize;
    int64_t end = std::min(begin + kBlockSize, numel);
    std::memcpy(&out.at(begin), &out_slices[blk_idx].at(0),
                (end - begin) * out.elsize());
  }

  return out;
}

ArrayRef _Lazy2B(KernelEvalContext* ctx, const ArrayRef& in) {
  if (in.eltype().isa<AShare>()) {
    return block_par_unary(ctx, "a2b", in);
  } else {
    SPU_ENFORCE(in.eltype().isa<BShare>());
    return in;
  }
}

ArrayRef _Lazy2A(KernelEvalContext* ctx, const ArrayRef& in) {
  if (in.eltype().isa<BShare>()) {
    return block_par_unary(ctx, "b2a", in);
  } else {
    SPU_ENFORCE(in.eltype().isa<AShare>(), "expect AShare, got {}",
                in.eltype());
    return in;
  }
}

// NOLINTBEGIN(bugprone-reserved-identifier)
#define _LAZY_AB ctx->getState<ABProtState>()->lazy_ab

#define _2A(x) _Lazy2A(ctx, x)
#define _2B(x) _Lazy2B(ctx, x)

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
#define _MulAA(lhs, rhs) block_par_binary(ctx, "mul_aa", lhs, rhs)
#define _MulA1B(lhs, rhs) block_par_binary(ctx, "mul_a1b", lhs, rhs)
#define _LShiftA(in, bits) ctx->caller()->call("lshift_a", in, bits)
#define _TruncA(in, bits) block_par_unary_with_size(ctx, "trunc_a", in, bits)
#define _MatMulAP(A, B, M, N, K) ctx->caller()->call("mmul_ap", A, B, M, N, K)
#define _MatMulAA(A, B, M, N, K) ctx->caller()->call("mmul_aa", A, B, M, N, K)
#define _B2P(x) ctx->caller()->call("b2p", x)
#define _P2B(x) ctx->caller()->call("p2b", x)
#define _A2B(x) block_par_unary(ctx, "a2b", x)
#define _B2A(x) block_par_unary(ctx, "b2a", x)
#define _NotB(x) ctx->caller()->call("not_b", x)
#define _AndBP(lhs, rhs) ctx->caller()->call("and_bp", lhs, rhs)
#define _AndBB(lhs, rhs) block_par_binary(ctx, "and_bb", lhs, rhs)
#define _XorBP(lhs, rhs) ctx->caller()->call("xor_bp", lhs, rhs)
#define _XorBB(lhs, rhs) ctx->caller()->call("xor_bb", lhs, rhs)
#define _LShiftB(in, bits) ctx->caller()->call("lshift_b", in, bits)
#define _RShiftB(in, bits) ctx->caller()->call("rshift_b", in, bits)
#define _ARShiftB(in, bits) ctx->caller()->call("arshift_b", in, bits)
#define _BitrevB(in, start, end) ctx->caller()->call("bitrev_b", in, start, end)
#define _MsbA(in) block_par_unary(ctx, "msb_a2b", in)
#define _RandA(size) ctx->caller()->call("rand_a", size)
#define _RandB(size) ctx->caller()->call("rand_b", size)
#define _EqualAP(lhs, rhs) block_par_binary(ctx, "equal_ap", lhs, rhs)
#define _EqualAA(lhs, rhs) block_par_binary(ctx, "equal_aa", lhs, rhs)

// NOLINTEND(bugprone-reserved-identifier)

class ABProtState : public State {
 public:
  static constexpr char kBindName[] = "ABProtState";

  bool lazy_ab = true;

  ABProtState() = default;
  explicit ABProtState(bool lazy) : lazy_ab(lazy) {}

  bool hasLowCostFork() const override { return true; }

  std::unique_ptr<State> fork() override {
    return std::make_unique<ABProtState>(lazy_ab);
  }
};

class ABProtCommonTypeS : public Kernel {
 public:
  static constexpr char kBindName[] = "common_type_s";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const Type& lhs = ctx->getParam<Type>(0);
    const Type& rhs = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

    if (lhs.isa<AShare>() && rhs.isa<AShare>()) {
      SPU_ENFORCE(lhs == rhs, "expect same, got lhs={}, rhs={}", lhs, rhs);
      ctx->setOutput(lhs);
    } else if (lhs.isa<AShare>() && rhs.isa<BShare>()) {
      ctx->setOutput(lhs);
    } else if (lhs.isa<BShare>() && rhs.isa<AShare>()) {
      ctx->setOutput(rhs);
    } else if (lhs.isa<BShare>() && rhs.isa<BShare>()) {
      ctx->setOutput(common_type_b(ctx->caller(), lhs, rhs));
    } else {
      SPU_THROW("should not be here, lhs={}, rhs={}", lhs, rhs);
    }
  }
};

class ABProtCastTypeS : public Kernel {
 public:
  static constexpr char kBindName[] = "cast_type_s";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const auto& frm = ctx->getParam<ArrayRef>(0);
    const auto& to_type = ctx->getParam<Type>(1);

    SPU_TRACE_MPC_DISP(ctx, frm, to_type);

    if (frm.eltype().isa<AShare>() && to_type.isa<AShare>()) {
      SPU_ENFORCE(frm.eltype() == to_type,
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
      SPU_THROW("should not be here, frm={}, to_type={}", frm, to_type);
    }
  }
};

class ABProtP2S : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2s";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_DISP(ctx, in);
    return _P2A(in);
  }
};

class ABProtS2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "s2p";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_DISP(ctx, in);
    if (_IsA(in)) {
      return _A2P(in);
    } else {
      SPU_ENFORCE(_IsB(in));
      return _B2P(in);
    }
  }
};

class ABProtRandS : public Kernel {
 public:
  static constexpr char kBindName[] = "rand_s";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const size_t size = ctx->getParam<size_t>(0);

    // ArrayRef proc(KernelEvalContext* ctx, size_t size) const override {
    SPU_TRACE_MPC_DISP(ctx, size);

    // always return random a share
    ctx->setOutput(_RandA(size));
  }
};

class ABProtNotS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_s";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_DISP(ctx, in);
    if (_LAZY_AB) {
      // TODO: Both A&B could handle not(invert).
      // if (in.eltype().isa<BShare>()) {
      //  return _NotB(in);
      //} else {
      //  SPU_ENFORCE(in.eltype().isa<AShare>());
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

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AddAP(_2A(lhs), rhs);
    }
    return _AddAP(lhs, rhs);
  }
};

class ABProtAddSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ss";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AddAA(_2A(lhs), _2A(rhs));
    }
    return _AddAA(lhs, rhs);
  }
};

class ABProtMulSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_sp";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _MulAP(_2A(lhs), rhs);
    }
    return _MulAP(lhs, rhs);
  }
};

class ABProtMulSS : public BinaryKernel {
  static bool hasMulA1B(KernelEvalContext* ctx) {
    return ctx->hasKernel("mul_a1b");
  }

 public:
  static constexpr char kBindName[] = "mul_ss";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

    if (hasMulA1B(ctx) && _IsA(rhs) && _IsB(lhs) && _NBits(lhs) == 1) {
      return _MulA1B(rhs, lhs);
    }
    if (hasMulA1B(ctx) && _IsA(lhs) && _IsB(rhs) && _NBits(rhs) == 1) {
      return _MulA1B(lhs, rhs);
    }
    if (_LAZY_AB) {
      // NOTE(juhou): Multiplication of two bits
      if (_IsB(lhs) && _NBits(lhs) == 1 && _IsB(rhs) && _NBits(rhs) == 1) {
        return _AndBB(lhs, rhs);
      }
      return _MulAA(_2A(lhs), _2A(rhs));
    }
    return _MulAA(lhs, rhs);
  }
};

class ABProtMatMulSP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_sp";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& a, const ArrayRef& b,
                size_t m, size_t n, size_t k) const override {
    SPU_TRACE_MPC_DISP(ctx, a, b);
    if (_LAZY_AB) {
      return _MatMulAP(_2A(a), b, m, n, k);
    }
    return _MatMulAP(a, b, m, n, k);
  }
};

class ABProtMatMulSS : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ss";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& a, const ArrayRef& b,
                size_t m, size_t n, size_t k) const override {
    SPU_TRACE_MPC_DISP(ctx, a, b);
    if (_LAZY_AB) {
      return _MatMulAA(_2A(a), _2A(b), m, n, k);
    }
    return _MatMulAA(a, b, m, n, k);
  }
};

class ABProtAndSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_sp";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AndBP(_2B(lhs), rhs);
    }
    return _B2A(_AndBP(_A2B(lhs), rhs));
  }
};

class ABProtAndSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_ss";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _AndBB(_2B(lhs), _2B(rhs));
    }
    return _B2A(_AndBB(_A2B(lhs), _A2B(rhs)));
  }
};

class ABProtXorSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_sp";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _XorBP(_2B(lhs), rhs);
    }
    return _B2A(_XorBP(_A2B(lhs), rhs));
  }
};

class ABProtXorSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_ss";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_TRACE_MPC_DISP(ctx, lhs, rhs);
    if (_LAZY_AB) {
      return _XorBB(_2B(lhs), _2B(rhs));
    }
    return _B2A(_XorBB(_A2B(lhs), _A2B(rhs)));
  }
};

class ABProtEqualSS : public Kernel {
 public:
  static constexpr char kBindName[] = "equal_ss";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const auto& x = ctx->getParam<ArrayRef>(0);
    const auto& y = ctx->getParam<ArrayRef>(1);

    SPU_TRACE_MPC_DISP(ctx, x, y);

    // TODO: use mandatory lazyAB pattern.
    SPU_ENFORCE(_LAZY_AB);

    ctx->setOutput(std::nullopt);

    // try fast path
    // TODO: use cost model instead of hand-coded priority.
    if (_IsA(x) && _IsA(y)) {
      if (ctx->hasKernel("equal_aa")) {
        ctx->setOutput(ctx->call("equal_aa", x, y));
      }
    } else if (_IsB(x) && _IsB(y)) {
      if (ctx->hasKernel("equal_bb")) {
        ctx->setOutput(ctx->call("equal_bb", x, y));
      }
    } else {
      // mixed A and B
      SPU_ENFORCE((_IsA(x) && _IsB(y)) || (_IsB(x) && _IsA(y)));

      if (ctx->hasKernel("equal_aa")) {
        ctx->setOutput(ctx->call("equal_aa", _2A(x), _2A(y)));
      }

      if (ctx->hasKernel("equal_bb")) {
        ctx->setOutput(ctx->call("equal_bb", _2B(x), _2B(y)));
      }
    }

    // fall through, no kernel found.
  }
};

class ABProtEqualSP : public Kernel {
 public:
  static constexpr char kBindName[] = "equal_sp";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    const auto& x = ctx->getParam<ArrayRef>(0);
    const auto& y = ctx->getParam<ArrayRef>(1);

    SPU_TRACE_MPC_DISP(ctx, x, y);

    // TODO: use mandatory lazyAB.
    SPU_ENFORCE(_LAZY_AB);

    ctx->setOutput(std::nullopt);

    if (_IsA(x) && ctx->hasKernel("equal_ap")) {
      ctx->setOutput(ctx->call("equal_ap", x, y));
    } else if (_IsB(x) && ctx->hasKernel("equal_bp")) {
      ctx->setOutput(ctx->call("equal_bp", x, y));
    }
  }
};

class ABProtLShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_s";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_DISP(ctx, in, bits);
    if (in.eltype().isa<AShare>()) {
      return _LShiftA(in, bits);
    } else if (in.eltype().isa<BShare>()) {
      if (_LAZY_AB) {
        return _LShiftB(in, bits);
      } else {
        return _B2A(_LShiftB(in, bits));
      }
    } else {
      SPU_THROW("Unsupported type {}", in.eltype());
    }
  }
};

class ABProtRShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_s";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_DISP(ctx, in, bits);
    if (_LAZY_AB) {
      return _RShiftB(_2B(in), bits);
    }
    return _B2A(_RShiftB(_A2B(in), bits));
  }
};

class ABProtARShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_s";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_DISP(ctx, in, bits);
    if (_LAZY_AB) {
      return _ARShiftB(_2B(in), bits);
    }
    return _B2A(_ARShiftB(_A2B(in), bits));
  }
};

class ABProtTruncS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "trunc_s";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_TRACE_MPC_DISP(ctx, in, bits);
    if (_LAZY_AB) {
      return _TruncA(_2A(in), bits);
    }
    return _TruncA(in, bits);
  }
};

class ABProtBitrevS : public Kernel {
 public:
  static constexpr char kBindName[] = "bitrev_s";

  Kind kind() const override { return Kind::Dynamic; }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<size_t>(1), ctx->getParam<size_t>(2)));
  }
  static ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                       size_t end) {
    SPU_TRACE_MPC_DISP(ctx, in, start, end);
    if (_LAZY_AB) {
      return _BitrevB(_2B(in), start, end);
    }
    return _B2A(_BitrevB(_A2B(in), start, end));
  }
};

class ABProtMsbS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_s";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_TRACE_MPC_DISP(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    if (ctx->hasKernel("msb_a2b")) {
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
};

}  // namespace

Type common_type_b(Object* ctx, const Type& a, const Type& b) {
  return ctx->call<Type>("common_type_b", a, b);
}

ArrayRef cast_type_b(Object* ctx, const ArrayRef& a, const Type& to_type) {
  return ctx->call("cast_type_b", a, to_type);
}

ArrayRef zero_a(Object* ctx, size_t sz) { return ctx->call("zero_a", sz); }

ArrayRef rand_a(Object* ctx, size_t sz) { return ctx->call("rand_a", sz); }

ArrayRef zero_b(Object* ctx, size_t sz) { return ctx->call("zero_b", sz); }

ArrayRef rand_b(Object* ctx, size_t sz) { return ctx->call("rand_b", sz); }

SPU_MPC_DEF_UNARY_OP(a2p)
SPU_MPC_DEF_UNARY_OP(p2a)
SPU_MPC_DEF_UNARY_OP(msb_a2b)
SPU_MPC_DEF_UNARY_OP(not_a)
SPU_MPC_DEF_BINARY_OP(add_ap)
SPU_MPC_DEF_BINARY_OP(add_aa)
SPU_MPC_DEF_BINARY_OP(mul_ap)
SPU_MPC_DEF_BINARY_OP(mul_aa)
SPU_MPC_DEF_BINARY_OP(mul_a1b)
SPU_MPC_DEF_UNARY_OP_WITH_SIZE(lshift_a)
SPU_MPC_DEF_UNARY_OP_WITH_SIZE(trunc_a)
SPU_MPC_DEF_MMUL(mmul_ap)
SPU_MPC_DEF_MMUL(mmul_aa)

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
SPU_MPC_DEF_UNARY_OP_WITH_2SIZE(bitrev_b);
SPU_MPC_DEF_BINARY_OP(add_bb)

ArrayRef bitintl_b(Object* ctx, const ArrayRef& in, size_t stride) {
  return ctx->call("bitintl_b", in, stride);
}

ArrayRef bitdeintl_b(Object* ctx, const ArrayRef& in, size_t stride) {
  return ctx->call("bitdeintl_b", in, stride);
}

void regABKernels(Object* obj) {
  obj->addState<ABProtState>();

  obj->regKernel<ABProtCommonTypeS>();
  obj->regKernel<ABProtCastTypeS>();
  obj->regKernel<ABProtP2S>();
  obj->regKernel<ABProtS2P>();
  obj->regKernel<ABProtRandS>();
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
  obj->regKernel<ABProtEqualSS>();
  obj->regKernel<ABProtEqualSP>();
  obj->regKernel<ABProtLShiftS>();
  obj->regKernel<ABProtRShiftS>();
  obj->regKernel<ABProtARShiftS>();
  obj->regKernel<ABProtTruncS>();
  obj->regKernel<ABProtBitrevS>();
  obj->regKernel<ABProtMsbS>();
}

}  // namespace spu::mpc
