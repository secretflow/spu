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

NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2,
                      FieldType field, bool has_mac) {
  SPU_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.shape() == s2.shape());

  const auto ty = makeType<AShrTy>(field, has_mac);
  SPU_ENFORCE(ty.size() == 2 * s1.elsize());
  NdArrayRef res(ty, s1.shape());

  auto res_s1 = getValueShare(res);
  auto res_s2 = getMacShare(res);

  ring_assign(res_s1, s1);
  ring_assign(res_s2, s2);
  return res;
}

NdArrayRef makeBShare(const NdArrayRef& s1, const NdArrayRef& s2,
                      FieldType field, int64_t nbits) {
  SPU_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.shape() == s2.shape());
  SPU_ENFORCE(s1.shape().back() % nbits == 0 && s1.shape().back() / nbits != 0);

  const PtType btype = calcBShareBacktype(nbits);
  const auto ty = makeType<BShrTy>(btype, nbits, field);
  const int64_t k = ty.as<BShrTy>()->k();
  SPU_ENFORCE(nbits <= k, "nbits = {}", nbits);

  Shape new_shape = s1.shape();
  new_shape.back() /= nbits;

  NdArrayRef res(ty, new_shape);
  int64_t res_numel = res.numel();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<std::array<ring2k_t, 2>> _res(res);

    pforeach(0, res_numel * k, [&](int64_t i) {
      _res[i][0] = 0;
      _res[i][1] = 0;
    });

    NdArrayView<ring2k_t> _s1(s1);
    NdArrayView<ring2k_t> _s2(s2);
    pforeach(0, res_numel, [&](int64_t i) {
      for (int64_t j = 0; j < nbits; ++j) {
        _res[i * k + j][0] = _s1[i * nbits + j];
        _res[i * k + j][1] = _s2[i * nbits + j];
      }
    });
  });
  return res;
}

NdArrayRef getShare(const NdArrayRef& in, int64_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1);

  if (in.eltype().isa<AShrTy>()) {
    Strides new_strides = in.strides();
    std::transform(new_strides.cbegin(), new_strides.cend(),
                   new_strides.begin(), [](int64_t s) { return 2 * s; });
    const auto field = in.eltype().as<AShrTy>()->field();
    const auto ty = makeType<RingTy>(field);
    return NdArrayRef{
        in.buf(), ty, in.shape(), new_strides,
        in.offset() + share_idx * static_cast<int64_t>(ty.size())};
  } else if (in.eltype().isa<BShrTy>()) {
    const auto field = in.eltype().as<BShrTy>()->field();
    const auto nbits = in.eltype().as<BShrTy>()->nbits();
    const auto k = in.eltype().as<BShrTy>()->k();
    const auto ty = makeType<RingTy>(field);

    Shape new_shape = in.shape();
    new_shape.back() *= nbits;

    if (nbits == k) {
      Strides new_strides = in.strides();
      std::transform(new_strides.cbegin(), new_strides.cend() - 1,
                     new_strides.begin(), [k](int64_t s) { return 2 * s * k; });
      new_strides.back() *= 2;
      return NdArrayRef{
          in.buf(), ty, new_shape, new_strides,
          in.offset() + share_idx * static_cast<int64_t>(ty.size())};
    } else {
      NdArrayRef ret(ty, new_shape);

      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        size_t numel = in.numel();
        NdArrayView<ring2k_t> _ret(ret);
        NdArrayView<std::array<ring2k_t, 2>> _in(in);

        pforeach(0, numel, [&](int64_t i) {
          for (size_t j = 0; j < nbits; ++j) {
            _ret[i * nbits + j] = _in[i * k + j][share_idx];
          }
        });
      });

      return ret;
    }
  } else {
    SPU_THROW("unsupported type {}", in.eltype());
  }
}

const NdArrayRef getValueShare(const NdArrayRef& in) { return getShare(in, 0); }

const NdArrayRef getMacShare(const NdArrayRef& in) { return getShare(in, 1); }

size_t maxNumBits(const NdArrayRef& lhs, const NdArrayRef& rhs) {
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
    return std::max(lhs.eltype().as<BShare>()->nbits(),
                    maxBitWidth<PShrT>(rhs));
  });
}

size_t minNumBits(const NdArrayRef& lhs, const NdArrayRef& rhs) {
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
    return std::min(lhs.eltype().as<BShare>()->nbits(),
                    maxBitWidth<PShrT>(rhs));
  });
}

// Convert a BShare in new_nbits
// then output only the values and macs of valid bits
std::pair<NdArrayRef, NdArrayRef> BShareSwitch2Nbits(const NdArrayRef& in,
                                                     int64_t new_nbits) {
  const int64_t old_nbits = in.eltype().as<BShrTy>()->nbits();
  if (old_nbits == new_nbits) {
    return {getValueShare(in), getMacShare(in)};
  }

  const int64_t p_num = in.numel();
  const auto field = in.eltype().as<Ring2k>()->field();
  auto out_shape = in.shape();
  out_shape.back() *= new_nbits;
  auto out_val = ring_zeros(field, out_shape);
  auto out_mac = ring_zeros(field, out_shape);

  auto in_val = getValueShare(in).clone();
  auto in_mac = getMacShare(in).clone();
  auto min_nbits = std::min(old_nbits, new_nbits);

  for (int64_t i = 0; i < p_num; ++i) {
    auto _in_val = NdArrayRef(in_val.buf(), makeType<RingTy>(field),
                              {min_nbits}, {1}, i * old_nbits * SizeOf(field));
    auto _in_mac = NdArrayRef(in_mac.buf(), makeType<RingTy>(field),
                              {min_nbits}, {1}, i * old_nbits * SizeOf(field));

    auto _out_val = NdArrayRef(out_val.buf(), makeType<RingTy>(field),
                               {min_nbits}, {1}, i * new_nbits * SizeOf(field));
    auto _out_mac = NdArrayRef(out_mac.buf(), makeType<RingTy>(field),
                               {min_nbits}, {1}, i * new_nbits * SizeOf(field));

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
