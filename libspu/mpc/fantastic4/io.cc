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

#include "libspu/mpc/fantastic4/io.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"
#include "libspu/core/context.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::fantastic4 {

Type Fantastic4Io::getShareType(Visibility vis, int owner_rank) const {
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(field_);
  } else if (vis == VIS_SECRET) {
    if (owner_rank >= 0 && owner_rank <= 3) {
      return makeType<Priv2kTy>(field_, owner_rank);
    } else {
      return makeType<AShrTy>(field_);
    }
  }
  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<NdArrayRef> Fantastic4Io::toShares(const NdArrayRef& raw, Visibility vis,
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
    if (owner_rank >= 0 && owner_rank <= 3) {
      // indicates private
      std::vector<NdArrayRef> shares;

      const auto ty = makeType<Priv2kTy>(field, owner_rank);
      for (int idx = 0; idx < 4; idx++) {
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

      SPU_ENFORCE(splits.size() == 4, "expect 4PC, got={}", splits.size());
      std::vector<NdArrayRef> shares;

      // Secret is split into 4 shares x_0, x_1, x_2, x_3
      // In our implementation, we let Party i (i in {0, 1, 2, 3}) holds x_i, x_i+1, x_i+2
      for (std::size_t i = 0; i < 4; i++) {
        shares.push_back(makeAShare(splits[i], splits[(i + 1) % 4], splits[(i + 2) % 4], field));
      }
      return shares;
    }
  }

  SPU_THROW("unsupported vis type {}", vis);
}

size_t Fantastic4Io::getBitSecretShareSize(size_t numel) const {
  const auto type = makeType<BShrTy>(PT_U8, 1);
  return numel * type.size();
}

std::vector<NdArrayRef> Fantastic4Io::makeBitSecret(const PtBufferView& in) const {
  PtType in_pt_type = in.pt_type;
  SPU_ENFORCE(in_pt_type == PT_I1);

  if (in_pt_type == PT_I1) {
    // we assume boolean is stored with byte array.
    in_pt_type = PT_U8;
  }

  const auto out_type = makeType<BShrTy>(PT_U8, /* out_nbits */ 1);
  const size_t numel = in.shape.numel();

  std::vector<NdArrayRef> shares = {NdArrayRef(out_type, in.shape),
                                    NdArrayRef(out_type, in.shape),
                                    NdArrayRef(out_type, in.shape),
                                    NdArrayRef(out_type, in.shape)};

  using bshr_el_t = uint8_t;
  using bshr_t = std::array<bshr_el_t, 3>;

  std::vector<bshr_el_t> r0(numel);
  std::vector<bshr_el_t> r1(numel);
  std::vector<bshr_el_t> r2(numel);

  yacl::crypto::PrgAesCtr(yacl::crypto::SecureRandSeed(), absl::MakeSpan(r0));
  yacl::crypto::PrgAesCtr(yacl::crypto::SecureRandSeed(), absl::MakeSpan(r1));
  yacl::crypto::PrgAesCtr(yacl::crypto::SecureRandSeed(), absl::MakeSpan(r2));

  NdArrayView<bshr_t> _s0(shares[0]);
  NdArrayView<bshr_t> _s1(shares[1]);
  NdArrayView<bshr_t> _s2(shares[2]);
  NdArrayView<bshr_t> _s3(shares[3]);
  
  // Secret is split into 4 shares x_0, x_1, x_2, x_3
  // In our implementation, we let Party i (i in {0, 1, 2, 3}) holds x_i, x_i+1, x_i+2
  for (size_t idx = 0; idx < numel; idx++) {
    const bshr_el_t r3 =
        static_cast<bshr_el_t>(in.get<bool>(idx)) ^ r0[idx] ^ r1[idx] ^ r2[idx];

    // P_0
    _s0[idx][0] = r0[idx] & 0x1;
    _s0[idx][1] = r1[idx] & 0x1;
    _s0[idx][2] = r2[idx] & 0x1;

    // P_1
    _s1[idx][0] = r1[idx] & 0x1;
    _s1[idx][1] = r2[idx] & 0x1;
    _s1[idx][2] = r3 & 0x1;

    // P_2
    _s2[idx][0] = r2[idx] & 0x1;
    _s2[idx][1] = r3 & 0x1;
    _s2[idx][2] = r0[idx] & 0x1;

    // P_3
    _s3[idx][0] = r3 & 0x1;
    _s3[idx][1] = r0[idx] & 0x1;
    _s3[idx][2] = r1[idx] & 0x1;
  }
  return shares;
}

NdArrayRef Fantastic4Io::fromShares(const std::vector<NdArrayRef>& shares) const {
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

    DISPATCH_ALL_FIELDS(field_, [&]() {
      using el_t = ring2k_t;
      using shr_t = std::array<el_t, 3>;

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
    NdArrayRef out(makeType<Pub2kTy>(field_), shares[0].shape());

    DISPATCH_ALL_FIELDS(field_, [&]() {
      NdArrayView<ring2k_t> _out(out);

      DISPATCH_UINT_PT_TYPES(eltype.as<BShrTy>()->getBacktype(), [&] {
        using shr_t = std::array<ScalarT, 3>;

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

std::unique_ptr<Fantastic4Io> makeFantastic4Io(FieldType field, size_t npc) {
  SPU_ENFORCE(npc == 4U, "fantastic4 is only for 4pc.");
  registerTypes();
  return std::make_unique<Fantastic4Io>(field, npc);
}

}