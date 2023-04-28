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

#include "libspu/mpc/cheetah/state.h"

#include "spdlog/spdlog.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

void CheetahMulState::makeSureCacheSize(FieldType field, int64_t numel) {
  // NOTE(juhou): make sure the lock is obtained
  SPU_ENFORCE(numel > 0);
  if (field_ != field) {
    // drop all previous cache
    cached_sze_ = 0;
  }

  if (cached_sze_ >= numel) {
    return;
  }

  //  create one batch OLE which then converted to Beavers
  //  Math:
  //   Alice samples rand0 and views it as rand0 = a0||b0
  //   Bob samples rand1 and views it as rand1 = b1||a0
  //   The multiplication rand0 * rand1 gives two cross term a0*b1||a1*b0
  //   Then the beaver (a0, b0, c0) and (a1, b1, c1)
  //   where c0 = a0*b0 + <a0*b1> + <a1*b0>
  //         c1 = a1*b1 + <a0*b1> + <a1*b0>
  const int rank = mul_prot_->Rank();
  const size_t ole_sze = mul_prot_->OLEBatchSize();
  const size_t num_ole = CeilDiv<size_t>(2 * numel, ole_sze);
  const size_t num_beaver = (num_ole * ole_sze) / 2;
  auto rand = ring_rand(field, num_ole * ole_sze);
  auto cross = mul_prot_->MulOLE(rand, rank == 0);

  ArrayRef beaver[3];
  ArrayRef a0b1, a1b0;
  if (rank == 0) {
    beaver[0] = rand.slice(0, num_beaver);
    beaver[1] = rand.slice(num_beaver, num_beaver * 2);
  } else {
    beaver[0] = rand.slice(num_beaver, num_beaver * 2);
    beaver[1] = rand.slice(0, num_beaver);
  }
  a0b1 = cross.slice(0, num_beaver);
  a1b0 = cross.slice(num_beaver, num_beaver * 2);

  beaver[2] = ring_add(ring_add(cross.slice(0, num_beaver),
                                cross.slice(num_beaver, 2 * num_beaver)),
                       ring_mul(beaver[0], beaver[1]));

  DISPATCH_ALL_FIELDS(field, "makeSureCacheSize", [&]() {
    for (size_t i : {0, 1, 2}) {
      auto tmp = ring_zeros(field, num_beaver + cached_sze_);
      ArrayView<const ring2k_t> old_cache(cached_beaver_[i]);
      ArrayView<const ring2k_t> _beaver(beaver[i]);
      ArrayView<ring2k_t> new_cache(tmp);
      // concate two array
      pforeach(0, cached_sze_, [&](int64_t j) { new_cache[j] = old_cache[j]; });
      pforeach(0, num_beaver,
               [&](int64_t j) { new_cache[cached_sze_ + j] = _beaver[j]; });
      cached_beaver_[i] = tmp;
    }
  });

  field_ = field;
  cached_sze_ += num_beaver;

  SPU_ENFORCE(cached_sze_ >= numel);
}

std::array<ArrayRef, 3> CheetahMulState::TakeCachedBeaver(FieldType field,
                                                          int64_t numel) {
  SPU_ENFORCE(numel > 0);
  std::unique_lock guard(lock_);
  makeSureCacheSize(field, numel);

  std::array<ArrayRef, 3> ret;
  for (size_t i : {0, 1, 2}) {
    SPU_ENFORCE(cached_beaver_[i].numel() >= numel);
    ret[i] = cached_beaver_[i].slice(0, numel);
    if (cached_sze_ == numel) {
      // empty cache now
      cached_beaver_[i] = ArrayRef();
    } else {
      cached_beaver_[i] = cached_beaver_[i].slice(numel, cached_sze_);
    }
  }

  cached_sze_ -= numel;
  return ret;
}

}  // namespace spu::mpc::cheetah
