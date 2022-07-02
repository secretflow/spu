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

#include "spu/mpc/semi2k/io.h"

namespace spu::mpc::cheetah {

class CheetahIo final : public semi2k::Semi2kIo {
 public:
  using semi2k::Semi2kIo::Semi2kIo;
};

std::unique_ptr<CheetahIo> makeCheetahIo(FieldType field, size_t npc);

}  // namespace spu::mpc::cheetah
