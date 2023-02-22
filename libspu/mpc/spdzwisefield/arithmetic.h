
#pragma once

#include "spu/mpc/kernel.h"
#include "spu/mpc/util/cexpr.h"

namespace spu::mpc::spdzwisefield {

using util::CExpr;
using util::Const;
using util::K;
using util::Log;
using util::N;

class P2A : public UnaryKernel {
    public:
        static constexpr char kBindName[] = "p2a";

        CExpr latency() const override { return Const(0); }

        CExpr comm() const override { return Const(0); }

        ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class A2P : public UnaryKernel {
    public:
        static constexpr char kBindName[] = "a2p";

        CExpr latency() const override { return Const(1); }

        CExpr comm() const override { return Const(0); }

        ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

}