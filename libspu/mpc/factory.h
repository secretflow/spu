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

#pragma once

#include <memory>

#include "yacl/link/link.h"

#include "libspu/core/context.h"
#include "libspu/mpc/io_interface.h"

#include "libspu/spu.pb.h"

namespace spu::mpc {

class Factory final {
 public:
  // Add a protocol to a context.
  //
  // @param config, a runtime config.
  // @param lctx, the inter party link context.
  static void RegisterProtocol(
      SPUContext* ctx, const std::shared_ptr<yacl::link::Context>& lctx);

  // Create a io context.
  //
  // @param kind, the protocol kind.
  // @param field, the working field.
  // @param npc, number of parties.
  //
  // Note: IO does not require a link context, especially for out-sourcing mode.
  static std::unique_ptr<IoInterface> CreateIO(const RuntimeConfig& conf,
                                               size_t npc);
};

}  // namespace spu::mpc
