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

#include "spu/mpc/aby3/io.h"

#include "yasl/crypto/pseudo_random_generator.h"
#include "yasl/utils/rand.h"

#include "spu/mpc/aby3/type.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {

std::vector<ArrayRef> Aby3Io::toShares(const ArrayRef& raw, Visibility vis,
                                       int owner_rank) const {
  YASL_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
               raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  YASL_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
               field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<ArrayRef>(world_size_, share);
  } else if (vis == VIS_SECRET) {
    // by default, make as arithmetic share.
    std::vector<ArrayRef> splits;
    if (owner_rank >= 0 && owner_rank <= 2) {
      // enable colocation optimization
      splits = ring_rand_additive_splits(raw, 2);
      size_t insertion_index = (owner_rank + 2) % 3;
      // currently, we have to use makeAShare to combine 2 array ref into 1
      // using 0-strided array become ill-defined.
      // One approach is to do compression at serialization level, we leave this
      // to later work so here we use ring_zeros instead of ring_zeros_packed
      // for aby3
      splits.insert(splits.begin() + insertion_index,
                    ring_zeros(field, raw.numel()));
    } else {
      splits = ring_rand_additive_splits(raw, world_size_);
    }
    YASL_ENFORCE(splits.size() == 3, "expect 3PC, got={}", splits.size());
    std::vector<ArrayRef> shares;
    for (std::size_t i = 0; i < 3; i++) {
      shares.push_back(
          makeAShare(splits[i], splits[(i + 1) % 3], field, owner_rank));
    }
    return shares;
  }

  YASL_THROW("unsupported vis type {}", vis);
}

std::vector<ArrayRef> Aby3Io::makeBitSecret(const ArrayRef& in) const {
  YASL_ENFORCE(in.eltype().isa<PtTy>(), "expected PtType, got {}", in.eltype());
  PtType in_pt_type = in.eltype().as<PtTy>()->pt_type();
  YASL_ENFORCE(in_pt_type == PT_BOOL);

  if (in_pt_type == PT_BOOL) {
    // we assume boolean is stored with byte array.
    in_pt_type = PT_U8;
  }

  const auto out_type = makeType<BShrTy>(PT_U8, /* out_nbits */ 1);
  const size_t numel = in.numel();
  constexpr auto kCryptoType = yasl::SymmetricCrypto::CryptoType::AES128_CTR;
  constexpr uint128_t kAesIV = 0U;

  std::vector<ArrayRef> shares{
      {out_type, numel}, {out_type, numel}, {out_type, numel}};
  return DISPATCH_UINT_PT_TYPES(in_pt_type, "_", [&]() {
    using InT = ScalarT;
    using BShrT = uint8_t;

    auto _in = ArrayView<InT>(in);

    std::vector<BShrT> r0(numel);
    std::vector<BShrT> r1(numel);

    uint64_t counter = 0;
    yasl::FillPseudoRandom(kCryptoType, yasl::RandSeed(), kAesIV, counter,
                           absl::MakeSpan(r0));
    yasl::FillPseudoRandom(kCryptoType, yasl::RandSeed(), kAesIV, counter,
                           absl::MakeSpan(r1));

    auto _s0 = ArrayView<std::array<BShrT, 2>>(shares[0]);
    auto _s1 = ArrayView<std::array<BShrT, 2>>(shares[1]);
    auto _s2 = ArrayView<std::array<BShrT, 2>>(shares[2]);

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      const BShrT r2 = static_cast<BShrT>(_in[idx]) - r0[idx] - r1[idx];

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

ArrayRef Aby3Io::fromShares(const std::vector<ArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();

  if (eltype.isa<Pub2kTy>()) {
    YASL_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    return shares[0].as(makeType<RingTy>(field_));
  } else if (eltype.isa<AShrTy>()) {
    YASL_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    ArrayRef out(makeType<Pub2kTy>(field_), shares[0].numel());
    DISPATCH_ALL_FIELDS(field_, "_", [&]() {
      auto _out = ArrayView<ring2k_t>(out);
      for (size_t si = 0; si < shares.size(); si++) {
        auto _share = ArrayView<std::array<ring2k_t, 2>>(shares[si]);
        for (auto idx = 0; idx < shares[0].numel(); idx++) {
          if (si == 0) {
            _out[idx] = 0;
          }
          _out[idx] += _share[idx][0];
        }
      }
    });
    return out;
  } else if (eltype.isa<BShrTy>()) {
    ArrayRef out(makeType<Pub2kTy>(field_), shares[0].numel());

    DISPATCH_ALL_FIELDS(field_, "_", [&]() {
      using OutT = ring2k_t;
      auto _out = ArrayView<OutT>(out);
      DISPATCH_UINT_PT_TYPES(eltype.as<BShrTy>()->getBacktype(), "_", [&] {
        using BShrT = ScalarT;
        for (size_t si = 0; si < shares.size(); si++) {
          auto _share = ArrayView<std::array<BShrT, 2>>(shares[si]);
          for (auto idx = 0; idx < shares[0].numel(); idx++) {
            if (si == 0) {
              _out[idx] = 0;
            }
            _out[idx] ^= _share[idx][0];
          }
        }
      });
    });

    return out;
  }
  YASL_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<Aby3Io> makeAby3Io(FieldType field, size_t npc) {
  YASL_ENFORCE_EQ(npc, 3u, "aby3 is only for 3pc.");
  registerTypes();
  return std::make_unique<Aby3Io>(field, npc);
}

}  // namespace spu::mpc::aby3
