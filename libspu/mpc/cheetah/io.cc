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

#include "libspu/mpc/cheetah/io.h"

#include "libspu/mpc/semi2k/type.h"

namespace spu::mpc::cheetah {

std::unique_ptr<CheetahIo> makeCheetahIo(FieldType field, size_t npc) {
  semi2k::registerTypes();

  return std::make_unique<CheetahIo>(field, npc);
}

}  // namespace spu::mpc::cheetah
