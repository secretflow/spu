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

#include <random>

#include "yasl/base/exception.h"
#include "yasl/crypto/pseudo_random_generator.h"

#include "spu/psi/cryptor/ecc_cryptor.h"

namespace spu {
class Sm2Cryptor : public IEccCryptor {
 public:
  explicit Sm2Cryptor(CurveType type = CurveType::CurveSm2) {
    std::random_device rd;
    yasl::PseudoRandomGenerator<uint64_t> prg(rd());

    prg.Fill(absl::MakeSpan(&private_key_[0], kEccKeySize));

    ec_group_nid_ = GetEcGroupId(type);
  }

  explicit Sm2Cryptor(absl::Span<const uint8_t> key,
                      CurveType type = CurveType::CurveSm2)
      : curve_type_(type) {
    YASL_ENFORCE(key.size() == kEccKeySize);
    std::memcpy(private_key_, key.data(), key.size());
    ec_group_nid_ = GetEcGroupId(type);
  }

  ~Sm2Cryptor() override { OPENSSL_cleanse(&private_key_[0], kEccKeySize); }

  void EccMask(absl::Span<const char> batch_points,
               absl::Span<char> dest_points) const override;

  CurveType GetCurveType() const override { return curve_type_; }

  size_t GetMaskLength() const override;

  std::vector<uint8_t> HashToCurve(absl::Span<const char> input) const override;

  static int GetEcGroupId(CurveType type) {
    switch (type) {
      case CurveType::CurveSecp256k1:
        return NID_secp256k1;
      case CurveType::CurveSm2:
        return NID_sm2;
      default:
        YASL_THROW("wron curve type:{}", static_cast<int>(type));
        return -1;
    }
  }

 private:
  CurveType curve_type_ = CurveType::CurveSm2;
  int ec_group_nid_ = NID_sm2;
};

}  // namespace spu