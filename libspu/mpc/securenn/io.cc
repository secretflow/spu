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

#include "libspu/mpc/securenn/io.h"

#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/securenn/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::securenn {

std::vector<ArrayRef> SecurennIo::toShares(const ArrayRef& raw, Visibility vis,
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
#if !defined(SPU_ENABLE_PRIVATE_TYPE)
    owner_rank = -1;
#endif

    if (owner_rank >= 0 && owner_rank < static_cast<int>(world_size_)) {
      // indicates private
      std::vector<ArrayRef> shares;
      const auto ty = makeType<Priv2kTy>(field, owner_rank);
      for (int idx = 0; idx < static_cast<int>(world_size_); idx++) {
        if (idx == owner_rank) {
          shares.push_back(raw.as(ty));
        } else {
          shares.push_back(makeConstantArrayRef(ty, raw.numel()));
        }
      }
      return shares;
    } else {
      // no colocation optmization
      std::vector<ArrayRef> shares;
      const auto ty = makeType<securenn::AShrTy>(field);

      const auto splits = ring_rand_additive_splits(raw, 2);
      for (const auto& split : splits) {
        shares.emplace_back(split.as(ty));
      }
      shares.emplace_back(ring_sub(splits[0], splits[0]).as(ty));
      return shares;
    }
  }
  SPU_THROW("unsupported vis type {}", vis);
}

ArrayRef SecurennIo::fromShares(const std::vector<ArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();
  const auto field = eltype.as<Ring2k>()->field();

  if (eltype.isa<Public>()) {
    return shares[0].as(makeType<RingTy>(field));
  } else if (eltype.isa<Secret>()) {
    ArrayRef res = ring_zeros(field, shares[0].numel());

    for (const auto& share : shares) {
      // Currently, only packed zeros are not compact, this is for colocation
      // optimization
      if (!share.isCompact()) {
        continue;
      }
      if (eltype.isa<AShare>()) {
        ring_add_(res, share);
      } else if (eltype.isa<BShare>()) {
        ring_xor_(res, share);
      } else {
        SPU_THROW("invalid share type {}", eltype);
      }
    }
    return res;
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<SecurennIo> makeSecurennIo(FieldType field, size_t npc) {
  registerTypes();
  return std::make_unique<SecurennIo>(field, npc);
}

}  // namespace spu::mpc::securenn
