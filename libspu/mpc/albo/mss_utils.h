#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/kernel.h"

// #define EQ_USE_OFFLINE
// #define EQ_USE_PRG_STATE

namespace spu::mpc::albo {
    NdArrayRef AssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                        const NdArrayRef& rhs);
    NdArrayRef RssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                        const NdArrayRef& rhs);
    NdArrayRef MssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                            const NdArrayRef& rhs);

    NdArrayRef RssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                        const NdArrayRef& rhs);
    NdArrayRef MssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                        const NdArrayRef& rhs);
    NdArrayRef MssAnd3NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                        const NdArrayRef& op2, const NdArrayRef& op3);
    NdArrayRef MssAnd4NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                        const NdArrayRef& op2, const NdArrayRef& op3, const NdArrayRef& op4);

    NdArrayRef ResharingAss2Rss(KernelEvalContext* ctx, const NdArrayRef& in);
    NdArrayRef ResharingAss2Mss(KernelEvalContext* ctx, const NdArrayRef& in);
    NdArrayRef ResharingRss2Mss(KernelEvalContext* ctx, const NdArrayRef& in);
    NdArrayRef ResharingMss2Rss(KernelEvalContext* ctx, const NdArrayRef& in);
    NdArrayRef ResharingMss2RssAri(KernelEvalContext* ctx, const NdArrayRef& in);
    NdArrayRef ResharingRss2Ass(KernelEvalContext* ctx, const NdArrayRef& in);
} // namespace spu::mpc::albo