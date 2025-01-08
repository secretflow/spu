
#pragma once

#include "yacl/link/link.h"

#include "libspu/core/context.h"

namespace spu::mpc {

std::unique_ptr<SPUContext> makeFantastic4Protocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx);

void regFantastic4Protocol(SPUContext* ctx,
                           const std::shared_ptr<yacl::link::Context>& lctx);

}  // namespace spu::mpc
