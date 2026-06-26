#ifndef SPU_MPC_FLP_FP_DIV_H_
#define SPU_MPC_FLP_FP_DIV_H_

#include "libspu/core/context.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/flp/float_type.h"

namespace spu::mpc::flp {

// Floating-point division: result = num / den
// Implements FFPDiv from SecFloat paper (Figure 6).
//
// exp_field: FieldType for exponent values (e.g., FM16 for p=8)
// man_field: FieldType for mantissa values (e.g., FM32 for q=23)
//
// Both inputs must have the same (p, q) parameters.
SharedFloat FPDiv(SPUContext* ctx, const SharedFloat& num,
                  const SharedFloat& den,
                  FieldType exp_field,
                  FieldType man_field);

}  // namespace spu::mpc::flp

#endif  // SPU_MPC_FLP_FP_DIV_H_
