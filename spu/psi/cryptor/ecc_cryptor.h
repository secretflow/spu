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

#include <array>
#include <cstring>
#include <memory>
#include <string_view>

#include "absl/types/span.h"
#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "yasl/base/exception.h"

namespace spu {

enum class CurveType {
  Curve25519,
  CurveFourQ,
  CurveSm2,
  CurveSecp256k1,
};

inline constexpr int kEccKeySize = 32;

// Make ECDH implementation plugable.
class IEccCryptor {
 public:
  IEccCryptor() {
    YASL_ENFORCE(RAND_bytes(&private_key_[0], kEccKeySize) == 1,
                 "Cannot create random private key");
  }

  virtual ~IEccCryptor() { OPENSSL_cleanse(&private_key_[0], kEccKeySize); }

  virtual void SetPrivateKey(absl::Span<const uint8_t> key) {
    YASL_ENFORCE(key.size() == kEccKeySize);
    std::memcpy(private_key_, key.data(), key.size());
  }

  /// Get current curve type
  virtual CurveType GetCurveType() const = 0;

  // Perform the ECC mask for a batch of items.
  //
  // The `base_points` contains a series of base point. Each point occupies
  // 256bit, i.e. 32 bytes.
  virtual void EccMask(absl::Span<const char> batch_points,
                       absl::Span<char> dest_points) const = 0;

  virtual size_t GetMaskLength() const { return kEccKeySize; }

  // Perform hash on input
  virtual std::vector<uint8_t> HashToCurve(absl::Span<const char> input) const;

 protected:
  uint8_t private_key_[kEccKeySize];
};

}  // namespace spu
