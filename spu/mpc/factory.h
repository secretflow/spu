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

#include "yasl/link/link.h"

#include "spu/mpc/io_interface.h"
#include "spu/mpc/object.h"

#include "spu/spu.pb.h"

namespace spu::mpc {

class Factory final {
 public:
  // Create a computation context with given link.
  //
  // @param kind, the protocol kind.
  // @param lctx, the inter party link context.
  static std::unique_ptr<Object> CreateCompute(
      ProtocolKind kind, const std::shared_ptr<yasl::link::Context>& lctx);

  // Create a io context.
  //
  // @param kind, the protocol kind.
  // @param field, the working field.
  // @param npc, number of parties.
  //
  // Note: IO does not require a link context, especially for out-sourcing mode.
  static std::unique_ptr<IoInterface> CreateIO(ProtocolKind kind,
                                               FieldType field, size_t npc);
};

}  // namespace spu::mpc
