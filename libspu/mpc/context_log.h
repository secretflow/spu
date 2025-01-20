#pragma once

#include <iostream>
#include "libspu/core/context.h"
#include "libspu/mpc/offline_recorder.h"
#include "libspu/mpc/common/communicator.h"

namespace spu::mpc {
void wrapStartRecorder(SPUContext* sctx);
void wrapStopRecorder(SPUContext* sctx);
void printStatus(SPUContext* sctx);
void clearStatus(SPUContext* sctx);
}  // namespace spu::mpc