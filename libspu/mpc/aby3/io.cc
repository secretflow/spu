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
#include "libspu/core/field_type_mapping.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {

Type Aby3Io::getShareType(Visibility vis, FieldType field,
                          int owner_rank) const {
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(field);
  } else if (vis == VIS_SECRET) {
    if (owner_rank >= 0 && owner_rank <= 2) {
      return makeType<Priv2kTy>(field, owner_rank);
    } else {
      return makeType<AShrTy>(field);
    }
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<NdArrayRef> Aby3Io::toShares(const NdArrayRef& raw, Visibility vis,
                                         int owner_rank) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();

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
  const auto type =
      makeType<BShrTy>(PT_U8, 1, getFieldFromPlainTextType(PT_U8));
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

  const auto out_type = makeType<BShrTy>(PT_U8, /* out_nbits */ 1,
                                         getFieldFromPlainTextType(PT_U8));
  const size_t numel = in.numel();

  std::vector<NdArrayRef> shares = {NdArrayRef(out_type, in.shape()),
                                    NdArrayRef(out_type, in.shape()),
                                    NdArrayRef(out_type, in.shape())};

  return DISPATCH_UINT_PT_TYPES(in_pt_type, "_", [&]() {
    using in_el_t = ScalarT;
    using bshr_el_t = uint8_t;
    using bshr_t = std::array<bshr_el_t, 2>;

    NdArrayView<in_el_t> _in(in);

    std::vector<bshr_el_t> r0(numel);
    std::vector<bshr_el_t> r1(numel);

    yacl::crypto::PrgAesCtr(yacl::crypto::RandSeed(), absl::MakeSpan(r0));
    yacl::crypto::PrgAesCtr(yacl::crypto::RandSeed(), absl::MakeSpan(r1));

    NdArrayView<bshr_t> _s0(shares[0]);
    NdArrayView<bshr_t> _s1(shares[1]);
    NdArrayView<bshr_t> _s2(shares[2]);

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      const bshr_el_t r2 = static_cast<bshr_el_t>(_in[idx]) - r0[idx] - r1[idx];

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
    return shares[0].as(makeType<RingTy>(eltype.as<Pub2kTy>()->field()));
  } else if (eltype.isa<Priv2kTy>()) {
    const size_t owner = eltype.as<Priv2kTy>()->owner();
    return shares[owner].as(makeType<RingTy>(eltype.as<Priv2kTy>()->field()));
  } else if (eltype.isa<AShrTy>()) {
    NdArrayRef out(makeType<Pub2kTy>(eltype.as<AShrTy>()->field()),
                   shares[0].shape());

    DISPATCH_ALL_FIELDS(eltype.as<AShrTy>()->field(), "_", [&]() {
      using el_t = ring2k_t;
      using shr_t = std::array<el_t, 2>;
      NdArrayView<ring2k_t> _out(out);
      for (size_t si = 0; si < shares.size(); si++) {
        NdArrayView<shr_t> _s(shares[si]);
        for (auto idx = 0; idx < shares[0].numel(); ++idx) {
          if (si == 0) {
            _out[idx] = 0;
          }
          _out[idx] += _s[idx][0];
        }
      }
    });
    return out;
  } else if (eltype.isa<BShrTy>()) {
    const auto* bt = eltype.as<BShrTy>();
    NdArrayRef out(makeType<Pub2kTy>(bt->getMappingField()), shares[0].shape());

    DISPATCH_ALL_FIELDS(bt->getMappingField(), "_", [&]() {
      NdArrayView<ring2k_t> _out(out);

      DISPATCH_UINT_PT_TYPES(eltype.as<BShrTy>()->getBacktype(), "_", [&] {
        using shr_t = std::array<ScalarT, 2>;
        for (size_t si = 0; si < shares.size(); si++) {
          NdArrayView<shr_t> _s(shares[si]);
          for (auto idx = 0; idx < shares[0].numel(); ++idx) {
            if (si == 0) {
              _out[idx] = 0;
            }
            _out[idx] ^= _s[idx][0];
          }
        }
      });
    });

    return out;
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<Aby3Io> makeAby3Io(size_t npc) {
  SPU_ENFORCE(npc == 3U, "aby3 is only for 3pc.");
  registerTypes();
  return std::make_unique<Aby3Io>(npc);
}

}  // namespace spu::mpc::aby3
