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

#include "libspu/mpc/spdz2k/io.h"

#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/spdz2k/value.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

std::vector<ArrayRef> Spdz2kIo::toShares(const ArrayRef& raw, Visibility vis,
                                         int owner_rank) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
              field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<ArrayRef>(world_size_, share);
  } else if (vis == VIS_SECRET) {
    // by default, make as arithmetic share.
    const auto zeros = ring_zeros(field, raw.numel());
    const auto splits = ring_rand_additive_splits(raw, world_size_);
    std::vector<ArrayRef> shares;
    shares.reserve(splits.size());
    for (const auto& split : splits) {
      shares.push_back(makeAShare(split, zeros, field));
    }
    return shares;
  }

  SPU_THROW("unsupported vis type {}", vis);
}

ArrayRef Spdz2kIo::fromShares(const std::vector<ArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();
  const auto field = eltype.as<Ring2k>()->field();

  if (eltype.isa<Public>()) {
    return shares[0].as(makeType<RingTy>(field));
  } else if (eltype.isa<Secret>()) {
    ArrayRef res = ring_zeros(field, shares.at(0).numel());
    for (const auto& share : shares) {
      if (eltype.isa<AShare>()) {
        ring_add_(res, getValueShare(share));
      } else if (eltype.isa<BShare>()) {
        ring_xor_(res, getValueShare(share));
      } else {
        SPU_THROW("invalid share type {}", eltype);
      }
    }
    return res;
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<Spdz2kIo> makeSpdz2kIo(FieldType field, size_t npc) {
  registerTypes();
  return std::make_unique<Spdz2kIo>(field, npc);
}

}  // namespace spu::mpc::spdz2k
