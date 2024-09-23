#pragma once

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::fantastic4 {

// Reference:

class A2B : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2b"; }

  ce::CExpr latency() const override {
    // 1 * AddBB : log(k) + 1
    // 1 * rotate: 1
    return Log(ce::K()) + 1 + 1;
  }

  // TODO: this depends on the adder circuit.
  ce::CExpr comm() const override {
    // 1 * AddBB : 2 * logk * k + k
    // 1 * rotate: k
    return 2 * Log(ce::K()) * ce::K() + ce::K() * 2;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

// class B2ASelector : public UnaryKernel {
//  public:
//   static constexpr const char* kBindName() { return "b2a"; }

//   Kind kind() const override { return Kind::Dynamic; }

//   NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
// };

class B2A : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }

  ce::CExpr latency() const override {
    // 2 * rotate   : 2
    // 1 * AddBB    : 1 + logk
    return ce::Const(3) + Log(ce::K());
  }

  // TODO: this depends on the adder circuit.
  ce::CExpr comm() const override {
    // 2 * rotate   : 2k
    // 1 * AddBB    : logk * k + k
    return Log(ce::K()) * ce::K() + 3 * ce::K();
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

// // Reference:
// // 5.4.1 Semi-honest Security
// // https://eprint.iacr.org/2018/403.pdf
// class B2AByOT : public UnaryKernel {
//  public:
//   static constexpr const char* kBindName() { return "b2a"; }

//   ce::CExpr latency() const override { return ce::Const(2); }

//   // Note: when nbits is large, OT method will be slower then circuit method.
//   ce::CExpr comm() const override {
//     return 2 * ce::K() * ce::K()  // the OT
//            + ce::K()              // partial send
//         ;
//   }

//   // FIXME: bypass unittest.
//   Kind kind() const override { return Kind::Dynamic; }

//   NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
// };

// class MsbA2B : public UnaryKernel {
//  public:
//   static constexpr const char* kBindName() { return "msb_a2b"; }

//   ce::CExpr latency() const override {
//     // 1 * carry : log(k) + 1
//     // 1 * rotate: 1
//     return Log(ce::K()) + 1 + 1;
//   }

//   ce::CExpr comm() const override {
//     // 1 * carry : k + 2 * k + 16 * 2
//     // 1 * rotate: k
//     return ce::K() + 2 * ce::K() + ce::K() + 32;
//   }

//   NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
// };

class EqualAA : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_aa"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};

class EqualAP : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "equal_ap"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override;
};


}  // namespace spu::mpc::fantastic4
