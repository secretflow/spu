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

  struct StateImpl {
    std::unique_ptr<spdz2k::Beaver> beaver_;
    // share of global key, share key has length of 128 bit
    uint128_t key_ = 0;
    // plaintext ring size, default set to half field bit length
    size_t k_ = 0;
    // statistical security parameter, default set to half field bit length
    size_t s_ = 0;

    StateImpl() = default;

    StateImpl(RuntimeConfig_BeaverType beaver_type,
              std::shared_ptr<yacl::link::Context> lctx, FieldType field) {
      if (beaver_type == RuntimeConfig_BeaverType_TrustedFirstParty) {
        beaver_ = std::make_unique<spdz2k::BeaverTfpUnsafe>(lctx);
      } else if (beaver_type == RuntimeConfig_BeaverType_MultiParty) {
        beaver_ = std::make_unique<spdz2k::BeaverTinyOt>(lctx);
      } else {
        SPU_THROW("unsupported beaver type {}", beaver_type);
      }
      auto runtime_field_ = getRuntimeField(field);
      k_ = SizeOf(field) * 8;
      s_ = k_;
      key_ = beaver_->InitSpdzKey(runtime_field_, s_);
    }

    spdz2k::Beaver* beaver() const { return beaver_.get(); }
    uint128_t key() const { return key_; }
    size_t k() const { return k_; }
    size_t s() const { return s_; }
  };

  std::pair<StateImpl, StateImpl> states_;

  std::shared_ptr<yacl::link::Context> lctx_;

 public:
  static constexpr char kBindName[] = "Spdz2kState";
  static constexpr auto kAesType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;

  static FieldType getRuntimeField(FieldType data_field, bool force = false) {
    switch (data_field) {
      case FM32:
        return FM64;
      case FM64:
        return FM128;
      case FM128: {
        if (force) {
          return FM128;
        }
        [[fallthrough]];
      }
      default:
        SPU_THROW("unsupported data field {} for spdz2k", data_field);
    }
    return FT_INVALID;
  }

  static FieldType getDataField(FieldType runtime_field) {
    switch (runtime_field) {
      case FM64:
        return FM32;
      case FM128:
        return FM64;
      default:
        SPU_THROW("unsupported data field {} for spdz2k", runtime_field);
    }
    return FT_INVALID;
  }

  explicit Spdz2kState(const RuntimeConfig& conf,
                       std::shared_ptr<yacl::link::Context> lctx) {
    // Init states for FM32
    lctx_ = lctx;
    states_ = {StateImpl(conf.beaver_type(), lctx, FM32),
               StateImpl(conf.beaver_type(), lctx, FM64)};
  }

  const StateImpl* getStateImpl(FieldType field) const {
    switch (field) {
      case FM64:
        return &states_.first;
      case FM128:
        return &states_.second;
      default:
        return nullptr;
    }
  }
};

}  // namespace spu::mpc
