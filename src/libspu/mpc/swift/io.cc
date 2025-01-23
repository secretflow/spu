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

#include "libspu/mpc/swift/io.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/core/context.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/swift/type.h"
#include "libspu/mpc/swift/value.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::swift {

Type SwiftIo::getShareType(Visibility vis, int owner_rank) const {
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(field_);
  } else if (vis == VIS_SECRET) {
    if (owner_rank >= 0 && owner_rank <= 2) {
      return makeType<Priv2kTy>(field_, owner_rank);
    } else {
      return makeType<AShrTy>(field_);
    }
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<NdArrayRef> SwiftIo::toShares(const NdArrayRef& raw, Visibility vis,
                                          int owner_rank) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
              field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<NdArrayRef>(world_size_, share);
  } else if (vis == VIS_SECRET) {
    if (owner_rank >= 0 && owner_rank <= 2) {
      // indicates private
      std::vector<NdArrayRef> shares;

      const auto ty = makeType<Priv2kTy>(field, owner_rank);
      for (int idx = 0; idx < 3; idx++) {
        if (idx == owner_rank) {
          shares.push_back(raw.as(ty));
        } else {
          shares.push_back(makeConstantArrayRef(ty, raw.shape()));
        }
      }
      return shares;
    } else {
      // normal secret
      SPU_ENFORCE(owner_rank == -1, "not a valid owner {}", owner_rank);

      // by default, make as arithmetic share.
      std::vector<NdArrayRef> shares;

      const auto alpha = ring_rand(field, raw.shape());
      // beta = raw + alpha
      const auto gamma = ring_rand(field, raw.shape());
      const auto beta = ring_add(raw, alpha);
      const auto gamma_plus_beta = ring_add(gamma, beta);
      const auto split_alpha = ring_rand_additive_splits(alpha, 2);

      // P0 : alpha_1,  alpha_2,  beta + gamma
      // P1 : alpha_1,  beta,     gamma
      // P2 : alpha_2,  beta,     gamma
      shares.push_back(
          makeAShare(split_alpha[0], split_alpha[1], gamma_plus_beta, field));

      shares.push_back(makeAShare(split_alpha[0], beta, gamma, field));

      shares.push_back(makeAShare(split_alpha[1], beta, gamma, field));

      return shares;
    }
  }

  SPU_THROW("unsupported vis type {}", vis);
}

NdArrayRef SwiftIo::fromShares(const std::vector<NdArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();

  if (eltype.isa<Pub2kTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field(), "field_={}, got={}",
                field_, eltype.as<Ring2k>()->field());
    return shares[0].as(makeType<RingTy>(field_));
  } else if (eltype.isa<Priv2kTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field(), "field_={}, got={}",
                field_, eltype.as<Ring2k>()->field());
    const size_t owner = eltype.as<Private>()->owner();
    return shares[owner].as(makeType<RingTy>(field_));
  } else if (eltype.isa<Secret>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field(), "field_={}, got={}",
                field_, eltype.as<Ring2k>()->field());
    NdArrayRef out(makeType<Pub2kTy>(field_), shares[0].shape());
    SPU_ENFORCE(shares.size() == 3, "expect shares.size()=3, got={}",
                shares.size());

    DISPATCH_ALL_FIELDS(field_, [&]() {
      using el_t = ring2k_t;
      using shr_t = std::array<el_t, 3>;
      NdArrayView<ring2k_t> _out(out);
      NdArrayView<shr_t> _s0(shares[0]);
      NdArrayView<shr_t> _s1(shares[1]);

      if (eltype.isa<AShare>()) {
        for (auto idx = 0; idx < shares[0].numel(); ++idx) {
          _out[idx] = _s1[idx][1] - _s0[idx][0] - _s0[idx][1];
        }
      } else if (eltype.isa<BShare>()) {
        for (auto idx = 0; idx < shares[0].numel(); ++idx) {
          _out[idx] = _s1[idx][1] ^ _s0[idx][0] ^ _s0[idx][1];
        }
      }
    });
    return out;
  }

  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<SwiftIo> makeSwiftIo(FieldType field, size_t npc) {
  SPU_ENFORCE(npc == 3U, "swift is only for 3pc.");
  registerTypes();
  return std::make_unique<SwiftIo>(field, npc);
}

}  // namespace spu::mpc::swift
