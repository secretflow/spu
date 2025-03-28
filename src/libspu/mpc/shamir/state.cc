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

#include "libspu/mpc/shamir/state.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/gfmp_ops.h"

namespace spu::mpc::shamir {

ShamirPrecomputedState::ShamirPrecomputedState(size_t _world_size,
                                               size_t _threshold)
    : world_size(_world_size), threshold(_threshold) {
  Vandermonde_n_by_n_minus_t_32 =
      GenVandermondeMatrix<uint32_t>(world_size, world_size - threshold);
  Vandermonde_n_by_n_minus_t_64 =
      GenVandermondeMatrix<uint64_t>(world_size, world_size - threshold);
  Vandermonde_n_by_n_minus_t_128 =
      GenVandermondeMatrix<uint128_t>(world_size, world_size - threshold);

  reconstruct_t_32 = GenReconstructVector<uint32_t>(_threshold + 1);
  reconstruct_t_64 = GenReconstructVector<uint64_t>(_threshold + 1);
  reconstruct_t_128 = GenReconstructVector<uint128_t>(_threshold + 1);
  reconstruct_2t_32 = GenReconstructVector<uint32_t>((_threshold << 1) + 1);
  reconstruct_2t_64 = GenReconstructVector<uint64_t>((_threshold << 1) + 1);
  reconstruct_2t_128 = GenReconstructVector<uint128_t>((_threshold << 1) + 1);
}

std::unique_ptr<State> ShamirPrecomputedState::fork() {
  auto new_shamir =
      std::make_unique<ShamirPrecomputedState>(world_size, threshold);
  return new_shamir;
}
}  // namespace spu::mpc::shamir