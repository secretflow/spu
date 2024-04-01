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

#include "yacl/crypto/rand/rand.h"
#include "yacl/link/link.h"

#include "libspu/core/object.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/spdz2k/beaver/beaver_tfp.h"
#include "libspu/mpc/spdz2k/beaver/beaver_tinyot.h"
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
  // #ifdef TINYOT
  //   using Beaver = spdz2k::BeaverTinyOt;
  // #else
  //   using Beaver = spdz2k::BeaverTfpUnsafe;
  // #endif

  std::unique_ptr<spdz2k::Beaver> beaver_;

  std::shared_ptr<yacl::link::Context> lctx_;

  // share of global key, share key has length of 128 bit
  uint128_t key_ = 0;

  // plaintext ring size, default set to half field bit length
  size_t k_ = 0;

  // statistical security parameter, default set to half field bit length
  size_t s_ = 0;

  FieldType data_field_ = FT_INVALID;

  FieldType runtime_field_ = FT_INVALID;

 private:
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

 public:
  static constexpr char kBindName[] = "Spdz2kState";
  static constexpr auto kAesType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;

  explicit Spdz2kState(const RuntimeConfig& conf,
                       std::shared_ptr<yacl::link::Context> lctx)
      : data_field_(conf.field()) {
    if (conf.beaver_type() == RuntimeConfig_BeaverType_TrustedFirstParty) {
      beaver_ = std::make_unique<spdz2k::BeaverTfpUnsafe>(lctx);
    } else if (conf.beaver_type() == RuntimeConfig_BeaverType_MultiParty) {
      beaver_ = std::make_unique<spdz2k::BeaverTinyOt>(lctx);
    } else {
      SPU_THROW("unsupported beaver type {}", conf.beaver_type());
    }
    lctx_ = lctx;
    runtime_field_ = getRuntimeField(data_field_);
    k_ = SizeOf(data_field_) * 8;
    s_ = k_;
    key_ = beaver_->InitSpdzKey(runtime_field_, s_);
  }

  FieldType getDefaultField() const { return runtime_field_; }

  spdz2k::Beaver* beaver() { return beaver_.get(); }

  uint128_t key() const { return key_; }

  size_t k() const { return k_; }

  size_t s() const { return s_; }
};

}  // namespace spu::mpc
