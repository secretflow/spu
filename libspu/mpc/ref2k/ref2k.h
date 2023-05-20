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

#include "yacl/link/link.h"

#include "libspu/core/context.h"
#include "libspu/mpc/io_interface.h"

namespace spu::mpc {

class Ref2kIo final : public BaseIo {
 public:
  using BaseIo::BaseIo;

  std::vector<ArrayRef> toShares(const ArrayRef& raw, Visibility vis,
                                 int owner_rank) const override;

  ArrayRef fromShares(const std::vector<ArrayRef>& shares) const override;
};

std::unique_ptr<SPUContext> makeRef2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx);
void regRef2kProtocol(SPUContext* ctx,
                      const std::shared_ptr<yacl::link::Context>& lctx);

std::unique_ptr<Ref2kIo> makeRef2kIo(FieldType field, size_t npc);

}  // namespace spu::mpc
