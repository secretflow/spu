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

#include "libspu/mpc/shamir/io.h"

#include "libspu/core/context.h"
#include "libspu/mpc/common/pv_gfmp.h"
#include "libspu/mpc/shamir/type.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::shamir {

Type ShamirIo::getShareType(Visibility vis, int owner_rank) const {
  if (vis == VIS_PUBLIC) {
    return makeType<PubGfmpTy>(field_);
  } else if (vis == VIS_SECRET) {
    SPU_ENFORCE(owner_rank == -1, "Private type is not supported");
    return makeType<AShrTy>(field_);
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<NdArrayRef> ShamirIo::toShares(const NdArrayRef& raw,
                                           Visibility vis,
                                           int owner_rank) const {
  SPU_ENFORCE(raw.eltype().isa<GfmpTy>(), "expected field type, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
              field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<PubGfmpTy>(field));
    return std::vector<NdArrayRef>(world_size_, share);
  } else if (vis == VIS_SECRET) {
    SPU_ENFORCE(owner_rank == -1, "private type is not supported");

    // by default, make as arithmetic share.
    std::vector<NdArrayRef> shares =
        gfmp_rand_shamir_shares(raw, world_size_, threshold_);

    for (size_t i = 0; i < shares.size(); ++i) {
      shares[i] = shares[i].as(makeType<AShrTy>(field));
    }
    return shares;
  }

  SPU_THROW("unsupported vis type {}", vis);
}

NdArrayRef ShamirIo::fromShares(const std::vector<NdArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();

  if (eltype.isa<PubGfmpTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    return shares[0].as(makeType<GfmpTy>(field_));
  } else if (eltype.isa<AShrTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    return gfmp_reconstruct_shamir_shares(shares, world_size_, threshold_);
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<ShamirIo> makeShamirIo(FieldType field, size_t npc,
                                       size_t threshold) {
  SPU_ENFORCE(npc >= threshold * 2 + 1 && threshold >= 1,
              "invalid party numbers {} or threshold {}", npc, threshold);
  registerTypes();
  return std::make_unique<ShamirIo>(field, npc, threshold);
}

}  // namespace spu::mpc::shamir
