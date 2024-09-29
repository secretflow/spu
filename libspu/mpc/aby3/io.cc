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

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/core/context.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {

Type Aby3Io::getShareType(Visibility vis, PtType type, int owner_rank) const {
  auto seman_type = GetEncodedType(type, field_);
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(seman_type);
  } else if (vis == VIS_SECRET) {
    if (owner_rank >= 0 && owner_rank <= 2) {
      return makeType<Priv2kTy>(seman_type, owner_rank);
    } else {
      return makeType<ArithShareTy>(seman_type, field_);
    }
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<MemRef> Aby3Io::toShares(const MemRef& raw, Visibility vis,
                                     int owner_rank) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());

  auto seman_type = raw.eltype().semantic_type();

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(seman_type));
    return std::vector<MemRef>(world_size_, share);
  } else if (vis == VIS_SECRET) {
    if (owner_rank >= 0 && owner_rank <= 2) {
      // indicates private
      std::vector<MemRef> shares;

      const auto ty = makeType<Priv2kTy>(seman_type, owner_rank);
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

      // Promote raw if it's smaller than field_
      MemRef raw_promoted;
      if (raw.elsize() * 8 != field_) {
        raw_promoted =
            MemRef(makeType<RingTy>(raw.eltype().semantic_type(), field_),
                   raw.shape());
        ring_assign(raw_promoted, raw);
      } else {
        raw_promoted = raw;
      }

      // by default, make as arithmetic share.
      std::vector<MemRef> splits =
          ring_rand_additive_splits(raw_promoted, world_size_);

      SPU_ENFORCE(splits.size() == 3, "expect 3PC, got={}", splits.size());
      std::vector<MemRef> shares;
      for (std::size_t i = 0; i < 3; i++) {
        shares.push_back(
            makeArithShare(splits[i], splits[(i + 1) % 3], seman_type, field_));
      }
      return shares;
    }
  }

  SPU_THROW("unsupported vis type {}", vis);
}

size_t Aby3Io::getBitSecretShareSize(size_t numel) const {
  const auto type = makeType<BoolShareTy>(SE_I8, ST_8, 1);
  return numel * type.size();
}

std::vector<MemRef> Aby3Io::makeBitSecret(const PtBufferView& in) const {
  PtType in_pt_type = in.pt_type;
  SPU_ENFORCE(in_pt_type == PT_I1);

  if (in_pt_type == PT_I1) {
    // we assume boolean is stored with byte array.
    in_pt_type = PT_U8;
  }

  const auto out_type = makeType<BoolShareTy>(SE_I8, ST_8, /* out_nbits */ 1);
  const size_t numel = in.shape.numel();

  std::vector<MemRef> shares = {MemRef(out_type, in.shape),
                                MemRef(out_type, in.shape),
                                MemRef(out_type, in.shape)};

  using bshr_el_t = uint8_t;
  using bshr_t = std::array<bshr_el_t, 2>;

  std::vector<bshr_el_t> r0(numel);
  std::vector<bshr_el_t> r1(numel);

  yacl::crypto::PrgAesCtr(yacl::crypto::SecureRandSeed(), absl::MakeSpan(r0));
  yacl::crypto::PrgAesCtr(yacl::crypto::SecureRandSeed(), absl::MakeSpan(r1));

  MemRefView<bshr_t> _s0(shares[0]);
  MemRefView<bshr_t> _s1(shares[1]);
  MemRefView<bshr_t> _s2(shares[2]);

  for (size_t idx = 0; idx < numel; idx++) {
    const bshr_el_t r2 =
        static_cast<bshr_el_t>(in.get<bool>(idx)) - r0[idx] - r1[idx];

    _s0[idx][0] = r0[idx] & 0x1;
    _s0[idx][1] = r1[idx] & 0x1;

    _s1[idx][0] = r1[idx] & 0x1;
    _s1[idx][1] = r2 & 0x1;

    _s2[idx][0] = r2 & 0x1;
    _s2[idx][1] = r0[idx] & 0x1;
  }
  return shares;
}

MemRef Aby3Io::fromShares(const std::vector<MemRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();

  auto seman_type = eltype.semantic_type();
  auto storage_type = eltype.storage_type();
  size_t field = SizeOf(storage_type) * 8;

  if (eltype.isa<Pub2kTy>()) {
    SPU_ENFORCE_GE(field_, field);
    return shares[0].as(makeType<RingTy>(seman_type, field));
  } else if (eltype.isa<Priv2kTy>()) {
    SPU_ENFORCE_GE(field_, field);
    const size_t owner = eltype.as<Private>()->owner();
    return shares[owner].as(makeType<RingTy>(seman_type, field));
  } else if (eltype.isa<ArithShareTy>()) {
    SPU_ENFORCE_GE(field_, field);
    MemRef out(makeType<RingTy>(seman_type, SizeOf(eltype.storage_type()) * 8),
               shares[0].shape());

    DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
      MemRefView<ScalarT> _out(out);

      for (size_t si = 0; si < shares.size(); si++) {
        DISPATCH_ALL_STORAGE_TYPES(shares[si].eltype().storage_type(), [&]() {
          MemRefView<std::array<ScalarT, 2>> _s(shares[si]);

          for (auto idx = 0; idx < shares[si].numel(); ++idx) {
            if (si == 0) {
              _out[idx] = 0;
            }
            _out[idx] += _s[idx][0];
          }
        });
      }
    });
    return out;
  } else if (eltype.isa<BoolShareTy>()) {
    MemRef out(makeType<Pub2kTy>(seman_type), shares[0].shape());

    DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
      MemRefView<ScalarT> _out(out);

      for (size_t si = 0; si < shares.size(); si++) {
        DISPATCH_ALL_STORAGE_TYPES(shares[si].eltype().storage_type(), [&]() {
          MemRefView<std::array<ScalarT, 2>> _s(shares[si]);

          for (auto idx = 0; idx < shares[0].numel(); ++idx) {
            if (si == 0) {
              _out[idx] = 0;
            }
            _out[idx] ^= _s[idx][0];
          }
        });
      }
    });

    return out;
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<Aby3Io> makeAby3Io(size_t field, size_t npc) {
  SPU_ENFORCE(npc == 3U, "aby3 is only for 3pc.");
  registerTypes();
  return std::make_unique<Aby3Io>(field, npc);
}

}  // namespace spu::mpc::aby3
