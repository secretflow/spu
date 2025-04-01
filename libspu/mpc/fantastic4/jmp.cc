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

#include "libspu/mpc/fantastic4/jmp.h"

namespace spu::mpc::fantastic4 {
  // compute previous party's rank
  size_t PrevRank(size_t rank, size_t world_size){
    return (rank + world_size -1) % world_size;
  }

  // compute rank offset from other party
  size_t OffsetRank(size_t myrank, size_t other, size_t world_size){
    size_t offset = (myrank + world_size -other) % world_size;
    if(offset == 3){
      offset = 1;
    }
    return offset;
  }
}