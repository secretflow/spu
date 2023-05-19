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

#include "libspu/mpc/common/pub2k.h"

#include <mutex>

#include "libspu/core/array_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

class Pub2kMakeP : public Kernel {
 public:
  static constexpr char kBindName[] = "make_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<uint128_t>(0), ctx->getParam<Shape>(1)));
  }

  static Value proc(KernelEvalContext* ctx, uint128_t init,
                    const Shape& shape) {
    const auto field = ctx->getState<Z2kState>()->getDefaultField();

    const auto eltype = makeType<Pub2kTy>(field);
    auto buf = std::make_shared<yacl::Buffer>(1 * eltype.size());
    NdArrayRef arr(buf,                                    // buffer
                   eltype,                                 // eltype
                   shape,                                  // shape
                   std::vector<int64_t>(shape.size(), 0),  // strides
                   0);

    DISPATCH_ALL_FIELDS(field, "pub2k.make_p", [&]() {
      arr.at<ring2k_t>(std::vector<int64_t>(shape.size(), 0)) =
          static_cast<ring2k_t>(init);
    });
    return Value(arr, DT_INVALID);
  }
};

class Pub2kRandP : public Kernel {
 public:
  static constexpr char kBindName[] = "rand_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<Shape>(0)));
  }

  static Value proc(KernelEvalContext* ctx, const Shape& shape) {
    auto* prg_state = ctx->getState<PrgState>();
    const auto field = ctx->getState<Z2kState>()->getDefaultField();

    auto arr =
        prg_state->genPubl(field, shape.numel()).as(makeType<Pub2kTy>(field));
    return WrapValue(arr, shape);
  }
};

class Pub2kNotP : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_not(in).as(makeType<Pub2kTy>(field));
  }
};

class Pub2kEqualPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override {
    SPU_ENFORCE_EQ(x.eltype(), y.eltype());
    SPU_ENFORCE(x.eltype().isa<Pub2kTy>());

    return ring_equal(x, y).as(x.eltype());
  }
};

class Pub2kAddPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kMulPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kMatMulPP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t m, size_t n,
                size_t k) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs, m, n, k).as(lhs.eltype());
  }
};

class Pub2kAndPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kXorPP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_pp";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Pub2kLShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class Pub2kRShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class Pub2kBitrevP : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    SPU_ENFORCE(start <= end);
    SPU_ENFORCE(end <= SizeOf(field) * 8);

    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class Pub2kARShiftP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class Pub2kTruncP : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "trunc_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    // Rounding
    // AxB = (AxB >> 14) + ((AxB >> 13) & 1);
    // See
    // https://stackoverflow.com/questions/14008330/how-do-you-multiply-two-fixed-point-numbers
    // Under certain pattern, like sum(mul(A, B)), error can accumulate in a
    // fairly significant way
    auto v1 = ring_arshift(in, bits);
    auto v2 = ring_arshift(in, bits - 1);
    ring_and_(v2, ring_ones(in.eltype().as<Ring2k>()->field(), in.numel()));
    ring_add_(v1, v2);
    return v1;
  }
};

class Pub2kMsbP : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_p";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return ring_rshift(in, in.elsize() * 8 - 1).as(in.eltype());
  }
};

void regPub2kTypes() {
  static std::once_flag flag;

  std::call_once(flag,
                 []() { TypeContext::getTypeContext()->addTypes<Pub2kTy>(); });
}

void regPub2kKernels(Object* obj) {
  obj->regKernel<Pub2kMakeP>();
  obj->regKernel<Pub2kRandP>();
  obj->regKernel<Pub2kNotP>();
  obj->regKernel<Pub2kEqualPP>();
  obj->regKernel<Pub2kAddPP>();
  obj->regKernel<Pub2kMulPP>();
  obj->regKernel<Pub2kMatMulPP>();
  obj->regKernel<Pub2kAndPP>();
  obj->regKernel<Pub2kXorPP>();
  obj->regKernel<Pub2kLShiftP>();
  obj->regKernel<Pub2kRShiftP>();
  obj->regKernel<Pub2kBitrevP>();
  obj->regKernel<Pub2kARShiftP>();
  obj->regKernel<Pub2kMsbP>();
  obj->regKernel<Pub2kTruncP>();
}

}  // namespace spu::mpc
