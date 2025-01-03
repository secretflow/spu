#pragma once

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/kernel.h"

// Only turn mask on in debug build
#ifndef NDEBUG
#define ENABLE_MASK_DURING_FANTASTIC4_P2A
#endif

namespace spu::mpc::fantastic4 {


class A2P : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2p"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class P2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2a"; }

  ce::CExpr latency() const override {
#ifdef ENABLE_MASK_DURING_FANTASTIC4_P2A
    return ce::Const(1);
#else
    return ce::Const(0);
#endif
  }

  ce::CExpr comm() const override {
#ifdef ENABLE_MASK_DURING_FANTASTIC4_P2A
    return ce::K();
#else
    return ce::Const(0);
#endif
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class A2V : public RevealToKernel {
 public:
  static constexpr const char* kBindName() { return "a2v"; }

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override {
    // 1 * send/recv: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  size_t rank) const override;
};

class V2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "v2a"; }

  // TODO: communication is unbalanced
  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    return ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};



class RandA : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override;
};

class NegateA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

// ////////////////////////////////////////////////////////////////////
// // add family
// ////////////////////////////////////////////////////////////////////
class AddAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class AddAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_aa"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
class MulAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class MulAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_aa"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // todo:
    // How to compute comm?
    return 2 * ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};



// ////////////////////////////////////////////////////////////////////
// // matmul family
// ////////////////////////////////////////////////////////////////////
class MatMulAP : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_aa"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    auto m = ce::Variable("m", "rows of lhs");
    auto n = ce::Variable("n", "cols of rhs");
    return ce::K() * m * n * 2;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class LShiftA : public ShiftKernel {
 public:
  static constexpr const char* kBindName() { return "lshift_a"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override;
};

class TruncAPr : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }

  ce::CExpr latency() const override { return ce::Const(3); }

  ce::CExpr comm() const override { return 4 * ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

}