#ifndef SPU_MPC_FLP_FP_MUL_H_
#define SPU_MPC_FLP_FP_MUL_H_

#include "libspu/core/context.h"
#include "libspu/mpc/flp/float_type.h"

namespace spu::mpc::flp {

// Floating-point multiplication over SharedFloat.
//
// Current representation convention:
//   - z: secret arithmetic Value, 0/1 zero flag
//   - s: secret arithmetic Value, 0/1 sign bit
//   - e: secret arithmetic Value, unbiased exponent
//   - m: secret arithmetic Value, explicit mantissa with q + 1 bits
//
// The implementation is expected to:
//   1. compute z = lhs.z OR rhs.z over arithmetic 0/1 shares;
//   2. compute s = lhs.s XOR rhs.s over arithmetic 0/1 shares;
//   3. compute e = lhs.e + rhs.e;
//   4. compute mantissa product with mixed-width multiplication;
//   5. call RoundMantissaAndCheckSharedApi;
//   6. call FPCheck.
SharedFloat FPMul(SPUContext* ctx, const SharedFloat& lhs,
                  const SharedFloat& rhs);

}  // namespace spu::mpc::flp

#endif  // SPU_MPC_FLP_FP_MUL_H_