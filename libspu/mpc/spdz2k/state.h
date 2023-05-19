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

#pragma once

#include <complex>
#include <vector>

#include "yacl/crypto/utils/rand.h"
#include "yacl/link/link.h"

#include "libspu/core/array_ref.h"
#include "libspu/core/object.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/spdz2k/beaver/beaver_tfp.h"
#include "libspu/mpc/spdz2k/commitment.h"

namespace spu::mpc {

template <FieldType _kField>
struct Spdz2kTrait {
  using element_t = typename Ring2kTrait<_kField>::scalar_t;
  constexpr static FieldType kField = _kField;
};

// re-use the std::complex definition for spdz2k share, where:
//   real(x) is the msg share piece.
//   imag(x) is the mac share piece.
template <typename T>
using Share = std::complex<T>;

class Spdz2kState : public State {
  std::unique_ptr<spdz2k::BeaverTfpUnsafe> beaver_;

  std::shared_ptr<yacl::link::Context> lctx_;

  // share of global key, share key has length of 128 bit
  uint128_t key_;

  // shares to be checked
  std::unique_ptr<std::vector<ArrayRef>> arr_ref_v_;

  // plaintext ring size
  const size_t k_ = 64;

  // statistical security parameter
  const size_t s_ = 64;

  // default in FM128
  const FieldType field_ = FM128;

 public:
  static constexpr char kBindName[] = "Spdz2kState";
  static constexpr auto kAesType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;

  explicit Spdz2kState(std::shared_ptr<yacl::link::Context> lctx) {
    beaver_ = std::make_unique<spdz2k::BeaverTfpUnsafe>(lctx);
    lctx_ = lctx;
    key_ = beaver_->GetSpdzKey(field_, s_);
    arr_ref_v_ = std::make_unique<std::vector<ArrayRef>>();
  }

  spdz2k::BeaverTfpUnsafe* beaver() { return beaver_.get(); }

  uint128_t key() const { return key_; }

  size_t k() const { return k_; }

  size_t s() const { return s_; }

  std::vector<ArrayRef>* arr_ref_v() { return arr_ref_v_.get(); }

  // public coin, used in malicious model, all party generate new seed, then
  // get exactly the same random variable.
  ArrayRef genPublCoin(FieldType field, size_t numel) {
    ArrayRef res(makeType<RingTy>(field), numel);

    // generate new seed
    uint128_t self_pk = yacl::crypto::RandSeed(true);
    std::vector<std::string> all_strs;

    std::string self_pk_str(reinterpret_cast<char*>(&self_pk), sizeof(self_pk));
    YACL_ENFORCE(commit_and_open(lctx_, self_pk_str, &all_strs));

    uint128_t public_seed = 0;
    for (const auto& str : all_strs) {
      uint128_t seed = *(reinterpret_cast<const uint128_t*>(str.data()));
      public_seed += seed;
    }

    yacl::crypto::FillPRand(
        kAesType, public_seed, 0, 0,
        absl::MakeSpan(static_cast<char*>(res.data()), res.buf()->size()));

    return res;
  }
};

}  // namespace spu::mpc
