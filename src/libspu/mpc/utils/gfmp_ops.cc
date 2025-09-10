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

#define PFOR_GRAIN_SIZE 4096

#include "libspu/mpc/utils/gfmp_ops.h"

#include <cstring>

#include "absl/types/span.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/linalg.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

#define SPU_ENFORCE_RING(x)                                           \
  SPU_ENFORCE((x).eltype().isa<Ring2k>(), "expect ring type, got={}", \
              (x).eltype());

#define SPU_ENFORCE_GFMP(x)                                           \
  SPU_ENFORCE((x).eltype().isa<GfmpTy>(), "expect gfmp type, got={}", \
              (x).eltype());

#define ENFORCE_EQ_ELSIZE_AND_SHAPE(lhs, rhs)                      \
  SPU_ENFORCE((lhs).elsize() == (rhs).elsize(),                    \
              "type size mismatch lhs={}, rhs={}", (lhs).eltype(), \
              (rhs).eltype());                                     \
  SPU_ENFORCE((lhs).shape() == (rhs).shape(),                      \
              "numel mismatch, lhs={}, rhs={}", lhs, rhs);

// Fast mod operation for Mersenne prime
void gfmp_mod_impl(NdArrayRef& ret, const NdArrayRef& x) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  const auto* ty = ret.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = mod_p(_x[idx]); });
  });
}

void gfmp_mul_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel,
             [&](int64_t idx) { _ret[idx] = mul_mod(_x[idx], _y[idx]); });
  });
}

void gfmp_add_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel,
             [&](int64_t idx) { _ret[idx] = add_mod(_x[idx], _y[idx]); });
  });
}

void gfmp_sub_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel, [&](int64_t idx) {
      _ret[idx] = add_mod(_x[idx], add_inv(_y[idx]));
    });
  });
}

void gfmp_div_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel, [&](int64_t idx) {
      _ret[idx] = mul_mod(_x[idx], mul_inv(_y[idx]));
    });
  });
}

}  // namespace
NdArrayRef gfmp_zeros(FieldType field, const Shape& shape) {
  NdArrayRef ret(makeType<GfmpTy>(field), shape);
  auto numel = ret.numel();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = 0; });
    return ret;
  });
}
NdArrayRef gfmp_rand(FieldType field, const Shape& shape) {
  uint64_t cnt = 0;
  return gfmp_rand(field, shape, yacl::crypto::SecureRandSeed(), &cnt);
}

// TODO(zjj): should fix it in yacl
// just make other unittest can be run, the implementation may not be correct.
namespace {

// these codes are copied from yacl/crypto/tools/prg.h
template <typename T>
struct IsSupportedMersennePrimeContainerType
    : public std::disjunction<
          std::is_same<uint128_t, T>, std::is_same<uint64_t, T>,
          std::is_same<uint32_t, T>, std::is_same<uint16_t, T>,
          std::is_same<uint8_t, T>> {};

template <typename T,
          std::enable_if_t<IsSupportedMersennePrimeContainerType<T>::value,
                           bool> = true>
constexpr T GetMersennePrimeMask() {
  if constexpr (std::is_same_v<T, uint128_t>) {
    return yacl::MakeUint128(std::numeric_limits<uint64_t>::max() >> 1,
                             std::numeric_limits<uint64_t>::max());
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return std::numeric_limits<uint64_t>::max() >> 3;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return std::numeric_limits<uint32_t>::max() >> 1;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return std::numeric_limits<uint16_t>::max() >> 3;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return std::numeric_limits<uint8_t>::max() >> 1;
  } else {
    YACL_THROW("Type T is not supported by FillPRandWithMersennePrime()");
  }
}

template <typename T,
          std::enable_if_t<IsSupportedMersennePrimeContainerType<T>::value,
                           bool> = true>
constexpr size_t GetMersennePrimeBitWidth() {
  if constexpr (std::is_same_v<T, uint128_t>) {
    return 127;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return 61;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return 31;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return 13;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return 7;
  } else {
    YACL_THROW("Type T is not supported by FillPRandWithMersennePrime()");
  }
}

template <typename T,
          std::enable_if_t<IsSupportedMersennePrimeContainerType<T>::value,
                           bool> = true>
T MersennePrimeMod(yacl::ByteContainerView buf) {
  YACL_ENFORCE(buf.size() ==
               sizeof(T) + (YACL_MODULE_SECPARAM_S_UINT("prg") + 7) / 8);
  YACL_ENFORCE((YACL_MODULE_SECPARAM_S_UINT("prg") + 7) / 8 < sizeof(uint64_t));

  constexpr auto k_mask = GetMersennePrimeMask<T>();
  // // using mpint::mod, expensive
  // math::MPInt rand;
  // rand.FromMagBytes(buf, Endian::little);
  // return rand.Mod(math::MPInt(k_mask)).Get<T>();

  // using native methods
  // buf should have 1 * T and 1 * s
  // | --- T-len --- | --- s-len --- |
  // lsb                         msb
  //
  // int i = k % p (where p = 2^s - 1) <= what we want
  // ---------------------------------
  // int i = (k & p) + (k >> s);
  // return (i >= p) ? i - p : i;
  //

  if constexpr (std::is_same_v<T, uint128_t> || std::is_same_v<T, uint64_t>) {
    T rand = 0;
    uint64_t aux_rand = 0;
    memcpy(&rand, buf.data(), sizeof(T));
    memcpy(&aux_rand, buf.data() + sizeof(T),
           (YACL_MODULE_SECPARAM_S_UINT("prg") + 7) / 8);

    // single round would work
    T i = (rand & k_mask) + aux_rand;
    return (i > k_mask) ? i - k_mask : i;
  } else {
    YACL_ENFORCE(buf.size() <= sizeof(uint128_t));
    uint128_t all_rand = 0;
    memcpy(&all_rand, buf.data(), buf.size());

    // constant round
    do {
      uint128_t i = (all_rand & k_mask) /* < 31 bit */ +
                    (all_rand >> GetMersennePrimeBitWidth<T>()) /* 40 bit */;
      all_rand = (i >= k_mask) ? i - k_mask : i;
    } while (all_rand >= k_mask);
    return (T)all_rand;
  }
}

template <typename T>
uint64_t FillPRandWithMersennePrimeTemp(
    yacl::crypto::SymmetricCrypto::CryptoType crypto_type, uint128_t seed,
    uint64_t iv, uint64_t count, absl::Span<T> out) {
  if constexpr (std::is_same_v<T, uint128_t> || std::is_same_v<T, uint64_t>) {
    // first, fill all outputs with randomness
    auto ret =
        yacl::crypto::FillPRand(crypto_type, seed, iv, count, (char*)out.data(),
                                out.size() * sizeof(T));

    // then, perform fast mod (in a non-standardized way)
    // NOTE: for mersenne prime with 127, 61 bit width, it's sufficient to
    // sample 127/61 bit uniform randomness directly, and then let the 2^127
    // value to be zero. Though this is not strictly uniform random, it will
    // provide statistical security of no less than 40 bits.
    constexpr auto k_mask = GetMersennePrimeMask<T>();
    for (auto& e : out) {
      e = (e & k_mask) == k_mask ? 0 : e & k_mask;
    }
    return ret;
  } else {
    // first, fill all outputs with randomness
    auto required_size =
        sizeof(T) + (YACL_MODULE_SECPARAM_S_UINT("prg") + 7) / 8;
    yacl::Buffer rand_bytes(out.size() * required_size);
    auto ret = yacl::crypto::FillPRand(crypto_type, seed, iv, count,
                                       (char*)rand_bytes.data(),
                                       out.size() * required_size);

    // then, perform mod
    yacl::ByteContainerView rand_view(rand_bytes);
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = MersennePrimeMod<T>(
          rand_view.subspan(i * required_size, required_size));
    }
    return ret;
  }
}
}  // namespace

NdArrayRef gfmp_rand(FieldType field, const Shape& shape, uint128_t prg_seed,
                     uint64_t* prg_counter) {
  constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
  constexpr uint128_t kAesInitialVector = 0U;
  NdArrayRef res(makeType<GfmpTy>(field), shape);
  DISPATCH_ALL_FIELDS(field, [&]() {
    *prg_counter = FillPRandWithMersennePrimeTemp<ring2k_t>(
        kCryptoType, prg_seed, kAesInitialVector, *prg_counter,
        absl::MakeSpan(&res.at<ring2k_t>(0), res.numel()));
  });
  return res;
}

NdArrayRef gfmp_mod(const NdArrayRef& x) {
  SPU_ENFORCE_GFMP(x);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_mod_impl(ret, x);
  return ret;
}

void gfmp_mod_(NdArrayRef& x) {
  SPU_ENFORCE_GFMP(x);
  gfmp_mod_impl(x, x);
}

NdArrayRef gfmp_mul_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_mul_mod_impl(ret, x, y);
  return ret;
}

void gfmp_mul_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_mul_mod_impl(x, x, y);
}

NdArrayRef gfmp_div_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_div_mod_impl(ret, x, y);
  return ret;
}

void gfmp_div_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_div_mod_impl(x, x, y);
}

NdArrayRef gfmp_add_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_add_mod_impl(ret, x, y);
  return ret;
}

void gfmp_add_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_add_mod_impl(x, x, y);
}

NdArrayRef gfmp_sub_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_sub_mod_impl(ret, x, y);
  return ret;
}

void gfmp_sub_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_sub_mod_impl(x, x, y);
}

// not requiring and not casting field.
void gfmp_exp_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel,
             [&](int64_t idx) { _ret[idx] = exp_mod(_x[idx], _y[idx]); });
  });
}

NdArrayRef gfmp_exp_mod(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_exp_mod_impl(ret, x, y);
  return ret;
}

void gfmp_exp_mod_(NdArrayRef& x, const NdArrayRef& y) {
  gfmp_exp_mod_impl(x, x, y);
}

}  // namespace spu::mpc
