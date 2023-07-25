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

#include "libspu/mpc/aby3/io.h"

#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/core/context.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {

Type Aby3Io::getShareType(Visibility vis, int owner_rank) const {
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

std::vector<NdArrayRef> Aby3Io::toShares(const NdArrayRef& raw, Visibility vis,
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
#if !defined(SPU_ENABLE_PRIVATE_TYPE)
    owner_rank = -1;
#endif

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
      std::vector<NdArrayRef> splits =
          ring_rand_additive_splits(raw, world_size_);

      SPU_ENFORCE(splits.size() == 3, "expect 3PC, got={}", splits.size());
      std::vector<NdArrayRef> shares;
      for (std::size_t i = 0; i < 3; i++) {
        shares.push_back(makeAShare(splits[i], splits[(i + 1) % 3], field));
      }
      return shares;
    }
  }

  SPU_THROW("unsupported vis type {}", vis);
}

size_t Aby3Io::getBitSecretShareSize(size_t numel) const {
  const auto type = makeType<BShrTy>(PT_U8, 1);
  return numel * type.size();
}

std::vector<NdArrayRef> Aby3Io::makeBitSecret(const NdArrayRef& in) const {
  SPU_ENFORCE(in.eltype().isa<PtTy>(), "expected PtType, got {}", in.eltype());
  PtType in_pt_type = in.eltype().as<PtTy>()->pt_type();
  SPU_ENFORCE(in_pt_type == PT_BOOL);

  if (in_pt_type == PT_BOOL) {
    // we assume boolean is stored with byte array.
    in_pt_type = PT_U8;
  }

  const auto out_type = makeType<BShrTy>(PT_U8, /* out_nbits */ 1);
  const size_t numel = in.numel();

  std::vector<NdArrayRef> shares = {NdArrayRef(out_type, in.shape()),
                                    NdArrayRef(out_type, in.shape()),
                                    NdArrayRef(out_type, in.shape())};

  return DISPATCH_UINT_PT_TYPES(in_pt_type, "_", [&]() {
    using InT = ScalarT;
    using BShrT = uint8_t;

    std::vector<BShrT> r0(numel);
    std::vector<BShrT> r1(numel);

    yacl::crypto::PrgAesCtr(yacl::crypto::RandSeed(), absl::MakeSpan(r0));
    yacl::crypto::PrgAesCtr(yacl::crypto::RandSeed(), absl::MakeSpan(r1));

    auto _s0 = shares[0].data<std::array<BShrT, 2>>();
    auto _s1 = shares[1].data<std::array<BShrT, 2>>();
    auto _s2 = shares[2].data<std::array<BShrT, 2>>();

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      const BShrT r2 = static_cast<BShrT>(in.at<InT>(idx)) - r0[idx] - r1[idx];

      _s0[idx][0] = r0[idx] & 0x1;
      _s0[idx][1] = r1[idx] & 0x1;

      _s1[idx][0] = r1[idx] & 0x1;
      _s1[idx][1] = r2 & 0x1;

      _s2[idx][0] = r2 & 0x1;
      _s2[idx][1] = r0[idx] & 0x1;
    }
    return shares;
  });
}

NdArrayRef Aby3Io::fromShares(const std::vector<NdArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();

  if (eltype.isa<Pub2kTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    return shares[0].as(makeType<RingTy>(field_));
  } else if (eltype.isa<Priv2kTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    const size_t owner = eltype.as<Private>()->owner();
    return shares[owner].as(makeType<RingTy>(field_));
  } else if (eltype.isa<AShrTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    NdArrayRef out(makeType<Pub2kTy>(field_), shares[0].shape());

    DISPATCH_ALL_FIELDS(field_, "_", [&]() {
      auto _out = out.data<ring2k_t>();
      for (size_t si = 0; si < shares.size(); si++) {
        for (auto idx = 0; idx < shares[0].numel(); ++idx) {
          if (si == 0) {
            _out[idx] = 0;
          }
          _out[idx] += shares[si].at<std::array<ring2k_t, 2>>(idx)[0];
        }
      }
    });
    return out;
  } else if (eltype.isa<BShrTy>()) {
    NdArrayRef out(makeType<Pub2kTy>(field_), shares[0].shape());

    DISPATCH_ALL_FIELDS(field_, "_", [&]() {
      using OutT = ring2k_t;
      auto _out = out.data<OutT>();

      DISPATCH_UINT_PT_TYPES(eltype.as<BShrTy>()->getBacktype(), "_", [&] {
        using BShrT = ScalarT;
        for (size_t si = 0; si < shares.size(); si++) {
          for (auto idx = 0; idx < shares[0].numel(); ++idx) {
            if (si == 0) {
              _out[idx] = 0;
            }
            _out[idx] ^= shares[si].at<std::array<BShrT, 2>>(idx)[0];
          }
        }
      });
    });

    return out;
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<Aby3Io> makeAby3Io(FieldType field, size_t npc) {
  SPU_ENFORCE(npc == 3U, "aby3 is only for 3pc.");
  registerTypes();
  return std::make_unique<Aby3Io>(field, npc);
}

}  // namespace spu::mpc::aby3
