#pragma once

#include "libspu/core/context.h"
#include "libspu/mpc/flp/float_type.h"

namespace spu::mpc::flp {

SharedFloat FPAdd(SPUContext* ctx, const SharedFloat& lhs,
                  const SharedFloat& rhs);

}  // namespace spu::mpc::flp
