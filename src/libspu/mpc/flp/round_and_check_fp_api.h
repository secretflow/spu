#ifndef SPU_MPC_FLP_ROUND_AND_CHECK_FP_API_H_
#define SPU_MPC_FLP_ROUND_AND_CHECK_FP_API_H_

#include <cstdint>

#include <memory>

#include "libspu/mpc/flp/float_type.h"
#include "libspu/mpc/api.h"

namespace spu::mpc::flp {

SharedFloat RoundMantissaAndCheckSharedApi(SPUContext* ctx,
                                           const SharedFloat& in, int Q,
                                           int q);

}  // namespace spu::mpc::flp

#endif  // SPU_MPC_FLP_ROUND_AND_CHECK_FP_API_H_