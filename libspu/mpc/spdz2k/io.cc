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

FieldType getRuntimeField(FieldType data_field) {
  switch (data_field) {
    case FM32:
      return FM64;
    case FM64:
      return FM128;
    default:
      SPU_THROW("unsupported data field {} for spdz2k", data_field);
  }
  return FT_INVALID;
}

Type Spdz2kIo::getShareType(Visibility vis, int /*owner_rank*/) const {
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(field_);
  } else if (vis == VIS_SECRET) {
    const auto runtime_field = getRuntimeField(field_);
    return makeType<AShrTy>(runtime_field);
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<NdArrayRef> Spdz2kIo::toShares(const NdArrayRef& raw,
                                           Visibility vis,
                                           int /*owner_rank*/) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
              field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<NdArrayRef>(world_size_, share);
  } else if (vis == VIS_SECRET) {
    const auto runtime_field = getRuntimeField(field);
    NdArrayRef x(makeType<Pub2kTy>(runtime_field), raw.shape());

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      NdArrayView<ring2k_t> _raw(raw);
      DISPATCH_ALL_FIELDS(runtime_field, "_", [&]() {
        NdArrayView<ring2k_t> _x(x);
        pforeach(0, raw.numel(), [&](int64_t idx) {
          _x[idx] = static_cast<ring2k_t>(_raw[idx]);
        });
      });
    });

    const auto zeros = ring_zeros(runtime_field, x.shape());
    const auto splits = ring_rand_additive_splits(x, world_size_);
    bool has_mac = false;
    std::vector<NdArrayRef> shares;
    shares.reserve(world_size_);
    for (const auto& split : splits) {
      // due to lack of information about key, MACs of data are set to zeros
      shares.push_back(makeAShare(split, zeros, runtime_field, has_mac));
    }
    return shares;
  }

  SPU_THROW("unsupported vis type {}", vis);
}

NdArrayRef Spdz2kIo::fromShares(const std::vector<NdArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();
  const auto field = eltype.as<Ring2k>()->field();

  if (eltype.isa<Public>()) {
    return shares[0].as(makeType<RingTy>(field));
  } else if (eltype.isa<Secret>()) {
    auto res = ring_zeros(field, shares.at(0).shape());
    for (const auto& share : shares) {
      if (eltype.isa<AShare>()) {
        ring_add_(res, getValueShare(share));
      } else if (eltype.isa<BShare>()) {
        ring_add_(res, getValueShare(share));
      } else {
        SPU_THROW("invalid share type {}", eltype);
      }
    }

    if (eltype.isa<AShare>()) {
      ring_bitmask_(res, 0, SizeOf(field_) * 8);
    } else {
      ring_bitmask_(res, 0, 1);
    }

    // TODO(zxp): use export_s to extract FM64 value from FM128
    {
      NdArrayRef x(makeType<Pub2kTy>(field_), res.shape());

      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        NdArrayView<ring2k_t> _res(res);
        DISPATCH_ALL_FIELDS(field_, "_", [&]() {
          NdArrayView<ring2k_t> _x(x);
          pforeach(0, x.numel(), [&](int64_t idx) {
            _x[idx] = static_cast<ring2k_t>(_res[idx]);
          });
        });
      });

      return x;
    }
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<Spdz2kIo> makeSpdz2kIo(FieldType field, size_t npc) {
  registerTypes();
  return std::make_unique<Spdz2kIo>(field, npc);
}

}  // namespace spu::mpc::spdz2k
