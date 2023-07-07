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

#include "libspu/mpc/spdz2k/value.h"

#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

ArrayRef makeAShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    bool has_mac) {
  SPU_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.numel() == s2.numel(), "s1 numel ={}, s2 numel ={}",
              s1.numel(), s2.numel());

  const auto ty = makeType<AShrTy>(field, has_mac);
  SPU_ENFORCE(ty.size() == 2 * s1.elsize());
  ArrayRef res(ty, s1.numel());

  auto res_s1 = getValueShare(res);
  auto res_s2 = getMacShare(res);

  ring_assign(res_s1, s1);
  ring_assign(res_s2, s2);
  return res;
}

ArrayRef makeBShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    size_t nbits) {
  SPU_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.numel() == s2.numel(), "s1 numel ={}, s2 numel ={}",
              s2.numel());
  SPU_ENFORCE(s1.numel() % nbits == 0 && s1.numel() / nbits != 0,
              "s1 numel = {}, nbits = {}", s1.numel(), nbits);

  const PtType btype = calcBShareBacktype(nbits);
  const auto ty = makeType<BShrTy>(btype, nbits, field);
  const size_t k = ty.as<BShrTy>()->k();
  SPU_ENFORCE(nbits <= k, "nbits = {}", nbits);

  ArrayRef res(ty, s1.numel() / nbits);
  size_t res_numel = res.numel();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto _res = ArrayView<std::array<ring2k_t, 2>>(res);
    auto _s1 = ArrayView<ring2k_t>(s1);
    auto _s2 = ArrayView<ring2k_t>(s2);

    pforeach(0, res_numel * k, [&](int64_t i) {
      _res[i][0] = 0;
      _res[i][1] = 0;
    });

    pforeach(0, res_numel, [&](int64_t i) {
      pforeach(0, nbits, [&](int64_t j) {
        _res[i * k + j][0] = _s1[i * nbits + j];
        _res[i * k + j][1] = _s2[i * nbits + j];
      });
    });
  });
  return res;
}

ArrayRef getShare(const ArrayRef& in, int64_t share_idx) {
  SPU_ENFORCE(in.stride() != 0);
  SPU_ENFORCE(share_idx == 0 || share_idx == 1);

  if (in.eltype().isa<AShrTy>()) {
    const auto field = in.eltype().as<AShrTy>()->field();
    const auto ty = makeType<RingTy>(field);
    return ArrayRef{in.buf(), ty, in.numel(), in.stride() * 2,
                    in.offset() + share_idx * static_cast<int64_t>(ty.size())};
  } else if (in.eltype().isa<BShrTy>()) {
    const auto field = in.eltype().as<BShrTy>()->field();
    const auto nbits = in.eltype().as<BShrTy>()->nbits();
    const auto k = in.eltype().as<BShrTy>()->k();
    const auto ty = makeType<RingTy>(field);

    if (nbits == k) {
      return ArrayRef{
          in.buf(), ty, in.numel() * static_cast<int64_t>(nbits),
          in.stride() * 2,
          in.offset() + share_idx * static_cast<int64_t>(ty.size())};
    } else {
      ArrayRef ret(ty, in.numel() * static_cast<int64_t>(nbits));

      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        auto _in = ArrayView<std::array<ring2k_t, 2>>(in);
        auto _ret = ArrayView<ring2k_t>(ret);
        size_t numel = in.numel();
        pforeach(0, numel, [&](int64_t i) {
          pforeach(0, nbits, [&](int64_t j) {
            _ret[i * nbits + j] = _in[i * k + j][share_idx];
          });
        });
      });

      return ret;
    }
  } else {
    SPU_THROW("unsupported type {}", in.eltype());
  }
}

const ArrayRef getValueShare(const ArrayRef& in) { return getShare(in, 0); }

const ArrayRef getMacShare(const ArrayRef& in) { return getShare(in, 1); }

size_t maxNumBits(const ArrayRef& lhs, const ArrayRef& rhs) {
  SPU_ENFORCE(lhs.eltype().isa<BShare>());
  SPU_ENFORCE(rhs.eltype().isa<BShare>() || rhs.eltype().isa<Public>());

  if (rhs.eltype().isa<BShare>()) {
    return std::max(lhs.eltype().as<BShare>()->nbits(),
                    rhs.eltype().as<BShare>()->nbits());
  }
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();
  const auto rhs_field = rhs_ty->field();
  return DISPATCH_ALL_FIELDS(rhs_field, "_", [&]() {
    using PShrT = ring2k_t;
    auto _rhs = ArrayView<PShrT>(rhs);
    return std::max(lhs.eltype().as<BShare>()->nbits(), maxBitWidth(_rhs));
  });
}

size_t minNumBits(const ArrayRef& lhs, const ArrayRef& rhs) {
  SPU_ENFORCE(lhs.eltype().isa<BShare>());
  SPU_ENFORCE(rhs.eltype().isa<BShare>() || rhs.eltype().isa<Public>());

  if (rhs.eltype().isa<BShare>()) {
    return std::min(lhs.eltype().as<BShare>()->nbits(),
                    rhs.eltype().as<BShare>()->nbits());
  }
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();
  const auto rhs_field = rhs_ty->field();
  return DISPATCH_ALL_FIELDS(rhs_field, "_", [&]() {
    using PShrT = ring2k_t;
    auto _rhs = ArrayView<PShrT>(rhs);
    return std::min(lhs.eltype().as<BShare>()->nbits(), maxBitWidth(_rhs));
  });
}

// Convert a BShare in new_nbits
// then output only the values and macs of valid bits
std::pair<ArrayRef, ArrayRef> BShareSwitch2Nbits(const ArrayRef& in,
                                                 size_t new_nbits) {
  const auto old_nbits = in.eltype().as<BShrTy>()->nbits();
  if (old_nbits == new_nbits) {
    return {getValueShare(in), getMacShare(in)};
  }

  // const size_t p_num = in.numel() / old_nbits;
  const size_t p_num = in.numel();
  const auto field = in.eltype().as<Ring2k>()->field();
  auto out_val = ring_zeros(field, p_num * new_nbits);
  auto out_mac = ring_zeros(field, p_num * new_nbits);

  auto in_val = getValueShare(in).clone();
  auto in_mac = getMacShare(in).clone();
  auto min_nbits = std::min(old_nbits, new_nbits);

  for (size_t i = 0; i < p_num; ++i) {
    auto _in_val = ArrayRef(in_val.buf(), makeType<RingTy>(field), min_nbits, 1,
                            i * old_nbits * SizeOf(field));
    auto _in_mac = ArrayRef(in_mac.buf(), makeType<RingTy>(field), min_nbits, 1,
                            i * old_nbits * SizeOf(field));

    auto _out_val = ArrayRef(out_val.buf(), makeType<RingTy>(field), min_nbits,
                             1, i * new_nbits * SizeOf(field));
    auto _out_mac = ArrayRef(out_mac.buf(), makeType<RingTy>(field), min_nbits,
                             1, i * new_nbits * SizeOf(field));

    ring_add_(_out_val, _in_val);
    ring_add_(_out_mac, _in_mac);
  }

  return {out_val, out_mac};
}

PtType calcBShareBacktype(size_t nbits) {
  if (nbits <= 8) {
    return PT_U8;
  }
  if (nbits <= 16) {
    return PT_U16;
  }
  if (nbits <= 32) {
    return PT_U32;
  }
  if (nbits <= 64) {
    return PT_U64;
  }
  if (nbits <= 128) {
    return PT_U128;
  }
  SPU_THROW("invalid number of bits={}", nbits);
}

}  // namespace spu::mpc::spdz2k
