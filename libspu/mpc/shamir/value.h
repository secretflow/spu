// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::shamir {

// The layout of Shamir Bshare.
//
// Shamir Bshare is a set of packed bit shares.
// Each bit share is 0/1 decoded on a field of Gfmp whose size is k. Nbits b
// shares are compacted to form an entire bshare.
//
//   element                    address
//   x[0].share_0               0
//   x[0].share_1               k
//   ...
//   x[0].share_(nbits-1)       (nbits-1)*k
//   ...
//   x[n-1].share_0             (n-1)*nbits*k+0
//   a[n-1].share_1             (n-1)*nbits*k+k
//

NdArrayRef getBitShare(const NdArrayRef& in, size_t bit_idx);

#define PFOR_GRAIN_SIZE 8192

}  // namespace spu::mpc::shamir
