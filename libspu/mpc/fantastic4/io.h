// Copyright 2025 Ant Group Co., Ltd.
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
#include "value.h"
namespace spu::mpc::fantastic4 {

class Fantastic4Io final : public BaseIo {
 public:
  using BaseIo::BaseIo;

  std::vector<NdArrayRef> toShares(const NdArrayRef& raw, Visibility vis, int owner_rank) const override;

  Type getShareType(Visibility vis, int owner_rank = -1) const override;

  NdArrayRef fromShares(const std::vector<NdArrayRef>& shares) const override;

  std::vector<NdArrayRef> makeBitSecret(const PtBufferView& in) const override;

  size_t getBitSecretShareSize(size_t numel) const override;

  bool hasBitSecretSupport() const override { return true; }
};

std::unique_ptr<Fantastic4Io> makeFantastic4Io(FieldType field, size_t npc);

} 