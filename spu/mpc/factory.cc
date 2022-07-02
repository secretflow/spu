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

#include "spu/mpc/factory.h"

#include <memory>

#include "yasl/base/exception.h"

#include "spu/mpc/aby3/io.h"
#include "spu/mpc/aby3/protocol.h"
#include "spu/mpc/cheetah/io.h"
#include "spu/mpc/cheetah/protocol.h"
#include "spu/mpc/ref2k/ref2k.h"
#include "spu/mpc/semi2k/io.h"
#include "spu/mpc/semi2k/protocol.h"

namespace spu::mpc {

std::unique_ptr<Object> Factory::CreateCompute(
    ProtocolKind kind, const std::shared_ptr<yasl::link::Context>& lctx) {
  switch (kind) {
    case ProtocolKind::REF2K: {
      return makeRef2kProtocol(lctx);
    }
    case ProtocolKind::SEMI2K: {
      return makeSemi2kProtocol(lctx);
    }
    case ProtocolKind::ABY3: {
      return makeAby3Protocol(lctx);
    }
    case ProtocolKind::CHEETAH: {
      return makeCheetahProtocol(lctx);
    }
    default: {
      YASL_THROW("Invalid protocol kind {}", kind);
    }
  }
  return nullptr;
}

std::unique_ptr<IoInterface> Factory::CreateIO(ProtocolKind kind,
                                               FieldType field, size_t npc) {
  switch (kind) {
    case ProtocolKind::REF2K: {
      return makeRef2kIo(field, npc);
    }
    case ProtocolKind::SEMI2K: {
      return semi2k::makeSemi2kIo(field, npc);
    }
    case ProtocolKind::ABY3: {
      return aby3::makeAby3Io(field, npc);
    }
    case ProtocolKind::CHEETAH: {
      return cheetah::makeCheetahIo(field, npc);
    }
    default: {
      YASL_THROW("Invalid protocol kind {}", kind);
    }
  }
  return nullptr;
}

}  // namespace spu::mpc
