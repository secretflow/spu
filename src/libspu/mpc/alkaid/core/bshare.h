#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/alkaid/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::alkaid::core {
/**
 * ----------------------------------------------
 *      Operation on POD type.
 * ----------------------------------------------
*/
template<typename PodT> using AssArray  = std::array<PodT, 2>;
template<typename PodT> using RssArray  = std::array<PodT, 2>;
template<typename PodT> using MrssArray = std::array<PodT, 3>;


template<typename LhsT, typename RhsT, typename OutT>
void AssXor2(const AssArray<LhsT>& lhs, const AssArray<RhsT>& rhs, AssArray<OutT>& out)
{
    out[0] = lhs[0] ^ rhs[0];
}

template<typename LhsT, typename RhsT, typename OutT>
void RssXor2(const RssArray<LhsT>& lhs, const RssArray<RhsT>& rhs, RssArray<OutT>& out)
{
    out[0] = lhs[0] ^ rhs[0];
    out[1] = lhs[1] ^ rhs[1];
}

template<typename LhsT, typename RhsT, typename OutT>
void MrssXor2(const MrssArray<LhsT>& lhs, const MrssArray<RhsT>& rhs, MrssArray<OutT>& out)
{
    out[0] = lhs[0] ^ rhs[0];
    out[1] = lhs[1] ^ rhs[1];
    out[2] = lhs[2] ^ rhs[2];
}

template<typename LhsT, typename RhsT, typename OutT>
void RssAnd2NoComm(
    const RssArray<LhsT>& lhs, const RssArray<RhsT>& rhs, 
    const OutT& r0, const OutT& r1,
    AssArray<OutT>& out) 
{
    out[0] = (lhs[0] & rhs[0]) ^ (lhs[0]  & rhs[1]) ^ 
             (lhs[1] & rhs[0]) ^ (r0 ^ r1);
}

template<typename LhsT, typename RhsT, typename OutT>
void MssAnd2NoComm(
    const MrssArray<LhsT>& lhs, const MrssArray<RhsT>& rhs, 
    const OutT& r0, const OutT& r1,
    RssArray<OutT>& out) 
{
    out[0] = (lhs[0] & rhs[0]) ^ (lhs[0] & rhs[1]) ^ 
             (lhs[1] & rhs[0]) ^ r0; 
    out[1] = (lhs[0] & rhs[0]) ^ (lhs[0] & rhs[2]) ^ 
             (lhs[2] & rhs[0]) ^ r1;
}

/**
 * ----------------------------------------------
 *      Operation on NdArrayRef.
 * ----------------------------------------------
*/
// Xor. Similiar to ABY3, the output bits number is aligned to the max bits number.
NdArrayRef AssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                   const NdArrayRef& rhs);
NdArrayRef RssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                   const NdArrayRef& rhs);
NdArrayRef MrssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                    const NdArrayRef& rhs);


NdArrayRef RssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs);
NdArrayRef MrssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                          const NdArrayRef& rhs);
NdArrayRef MrssAnd3NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                          const NdArrayRef& op2, const NdArrayRef& op3);
NdArrayRef MrssAnd4NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                          const NdArrayRef& op2, const NdArrayRef& op3,
                          const NdArrayRef& op4);

NdArrayRef ResharingAss2Rss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingAss2Mrss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingRss2Mrss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingMrss2Rss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingMrss2RssAri(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingRss2Ass(KernelEvalContext* ctx, const NdArrayRef& in);
} // namespace spu::mpc::alkaid::utils