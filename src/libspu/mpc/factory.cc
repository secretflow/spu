// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/factory.h"

#include <memory>

#include "libspu/core/prelude.h"
#include "libspu/mpc/aby3/io.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/cheetah/io.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/ref2k/ref2k.h"
#include "libspu/mpc/securenn/io.h"
#include "libspu/mpc/securenn/protocol.h"
#include "libspu/mpc/semi2k/io.h"
#include "libspu/mpc/semi2k/protocol.h"

namespace spu::mpc {

void Factory::RegisterProtocol(
    SPUContext* ctx, const std::shared_ptr<yacl::link::Context>& lctx) {
  // TODO: support multi-protocols.
  switch (ctx->config().protocol()) {
    case ProtocolKind::REF2K: {
      return regRef2kProtocol(ctx, lctx);
    }
    case ProtocolKind::SEMI2K: {
      return regSemi2kProtocol(ctx, lctx);
    }
    case ProtocolKind::ABY3: {
      return regAby3Protocol(ctx, lctx);
    }
    case ProtocolKind::CHEETAH: {
      return regCheetahProtocol(ctx, lctx);
    }
    case ProtocolKind::SECURENN: {
      return regSecurennProtocol(ctx, lctx);
    }
    default: {
      SPU_THROW("Invalid protocol kind {}", ctx->config().protocol());
    }
  }
}

std::unique_ptr<IoInterface> Factory::CreateIO(const RuntimeConfig& conf,
                                               size_t npc) {
  switch (conf.protocol()) {
    case ProtocolKind::REF2K: {
      return makeRef2kIo(conf.field(), npc);
    }
    case ProtocolKind::SEMI2K: {
      return semi2k::makeSemi2kIo(conf.field(), npc);
    }
    case ProtocolKind::ABY3: {
      return aby3::makeAby3Io(conf.field(), npc);
    }
    case ProtocolKind::CHEETAH: {
      return cheetah::makeCheetahIo(conf.field(), npc);
    }
    case ProtocolKind::SECURENN: {
      return securenn::makeSecurennIo(conf.field(), npc);
    }
    default: {
      SPU_THROW("Invalid protocol kind {}", conf.protocol());
    }
  }
  return nullptr;
}

}  // namespace spu::mpc
