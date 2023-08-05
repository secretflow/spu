#pragma once
#include "yacl/link/link.h"

#include "libspu/pir/pir.pb.h"

namespace spu::pir {
PirResultReport TrivialSpirFullyOnlineServer(
    const std::shared_ptr<yacl::link::Context>& lctx,
    const PirSetupConfig& config);

PirResultReport TrivialSpirFullyOnlineClient(
    const std::shared_ptr<yacl::link::Context>& lctx,
    const PirClientConfig& config);
}
