
#pragma once

#include "libspu/mpc/kernel.h"
#include "libspu/core/array_ref.h"

namespace spu::mpc::spdzwisefield {

class P2A : public UnaryKernel {
    public:
        static constexpr char kBindName[] = "p2a";

/*
 * Note that verification is needed for checking the correctness
 * of mac, but it is not included in the latency and comm.
 */
        ce::CExpr latency() const override {
            // one for pass around rss share and 
            // one for mac_key * share

            return ce::Const(2); 
        }

        ce::CExpr comm() const override {

            return ce::K() * ce::Const(2); 
        }

        ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class A2P : public UnaryKernel {
    public:
        static constexpr char kBindName[] = "a2p";

        ce::CExpr latency() const override { return ce::Const(1); }

        ce::CExpr comm() const override { return ce::K(); }

        ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

}