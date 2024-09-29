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

#include "libspu/mpc/io_interface.h"

namespace spu::mpc::aby3 {

class Aby3Io final : public BaseIo {
 public:
  using BaseIo::BaseIo;

  std::vector<MemRef> toShares(const MemRef& raw, Visibility vis,
                               int owner_rank) const override;

  Type getShareType(Visibility vis, PtType type,
                    int owner_rank = -1) const override;

  MemRef fromShares(const std::vector<MemRef>& shares) const override;

  std::vector<MemRef> makeBitSecret(const PtBufferView& in) const override;

  size_t getBitSecretShareSize(size_t numel) const override;

  bool hasBitSecretSupport() const override { return true; }
};

std::unique_ptr<Aby3Io> makeAby3Io(size_t field, size_t npc);

}  // namespace spu::mpc::aby3
