#pragma once

#include "libspu/core/context.h"
#include "libspu/mpc/flp/float_type.h"

namespace spu::mpc::flp {

SharedFloat FPCheck(SPUContext* ctx, const SharedFloat& sf);

}  // namespace spu::mpc::flp
