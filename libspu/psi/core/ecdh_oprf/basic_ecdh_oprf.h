// Copyright 2022 Ant Group Co., Ltd.
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
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/base/hash/hash_interface.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf.h"
#include "libspu/psi/cryptor/ecc_cryptor.h"
#include "libspu/psi/cryptor/sm2_cryptor.h"

// 2HashDH Oprf
// F_k(x) = H2(x, H1(x)^k)
// reference: JKK14
// Round-optimal password-protected secret sharing and T-PAKE in the
// password-only model
//   https://eprint.iacr.org/2014/650.pdf
//
// server private key: sk
// server  H2(x, H1(x)^sk)
// client  H2(y, (H1(y)^r)^sk^(1/r))=H2(y, H1(y)^sk)

namespace spu::psi {

class BasicEcdhOprfServer : public IEcdhOprfServer {
 public:
  BasicEcdhOprfServer() = default;

  /**
   * @brief Construct a new Basic Ecdh Oprf Server object
   *
   * @param type support CurveSecp256k1/Sm2/FourQ
   */
  explicit BasicEcdhOprfServer(CurveType type)
      : curve_type_(type), ec_group_nid_(Sm2Cryptor::GetEcGroupId(type)) {
    (void)curve_type_;
  }

  BasicEcdhOprfServer(yacl::ByteContainerView private_key, CurveType type)
      : IEcdhOprfServer(private_key),
        curve_type_(type),
        ec_group_nid_(Sm2Cryptor::GetEcGroupId(type)) {}

  ~BasicEcdhOprfServer() override = default;

  OprfType GetOprfType() const override { return OprfType::Basic; }

  std::string Evaluate(absl::string_view blinded_element) const override;

  std::string FullEvaluate(yacl::ByteContainerView input) const override;
  std::string SimpleEvaluate(yacl::ByteContainerView input) const override;

  size_t GetCompareLength() const override;
  size_t GetEcPointLength() const override;

  void SetHashType(yacl::crypto::HashAlgorithm hash_type) {
    hash_type_ = hash_type;
  }

 private:
  CurveType curve_type_;
  int ec_group_nid_{};
  yacl::crypto::HashAlgorithm hash_type_ = yacl::crypto::HashAlgorithm::BLAKE3;
};

class BasicEcdhOprfClient : public IEcdhOprfClient {
 public:
  explicit BasicEcdhOprfClient(CurveType type);
  BasicEcdhOprfClient(CurveType type, yacl::ByteContainerView private_key);
  BasicEcdhOprfClient(CurveType type, yacl::ByteContainerView private_key,
                      yacl::ByteContainerView private_key_inv);

  ~BasicEcdhOprfClient() override = default;

  OprfType GetOprfType() const override { return OprfType::Basic; }

  std::string Blind(absl::string_view input) const override;

  std::string Finalize(absl::string_view item,
                       absl::string_view evaluated_element) const override;

  std::string Finalize(absl::string_view evaluated_element) const override;

  size_t GetCompareLength() const override;
  size_t GetEcPointLength() const override;

  std::string Unblind(absl::string_view input) const override;

  void SetHashType(yacl::crypto::HashAlgorithm hash_type) {
    hash_type_ = hash_type;
  }

 private:
  CurveType curve_type_;
  int ec_group_nid_;

  std::vector<uint8_t> sk_inv_;

  yacl::crypto::HashAlgorithm hash_type_ = yacl::crypto::HashAlgorithm::BLAKE3;
};

class FourQBasicEcdhOprfServer : public IEcdhOprfServer {
 public:
  FourQBasicEcdhOprfServer() = default;

  explicit FourQBasicEcdhOprfServer(yacl::ByteContainerView private_key)
      : IEcdhOprfServer(private_key) {}

  ~FourQBasicEcdhOprfServer() override = default;

  OprfType GetOprfType() const override { return OprfType::Basic; }

  std::string Evaluate(absl::string_view blinded_element) const override;

  std::string FullEvaluate(yacl::ByteContainerView input) const override;
  std::string SimpleEvaluate(yacl::ByteContainerView input) const override;

  size_t GetCompareLength() const override;
  size_t GetEcPointLength() const override;

  void SetHashType(yacl::crypto::HashAlgorithm hash_type) {
    hash_type_ = hash_type;
  }

 private:
  yacl::crypto::HashAlgorithm hash_type_ = yacl::crypto::HashAlgorithm::BLAKE3;
};

class FourQBasicEcdhOprfClient : public IEcdhOprfClient {
 public:
  FourQBasicEcdhOprfClient();
  explicit FourQBasicEcdhOprfClient(yacl::ByteContainerView private_key);

  ~FourQBasicEcdhOprfClient() override = default;

  OprfType GetOprfType() const override { return OprfType::Basic; }

  std::string Blind(absl::string_view input) const override;

  std::string Finalize(absl::string_view item,
                       absl::string_view evaluated_element) const override;

  std::string Finalize(absl::string_view evaluated_element) const override;

  size_t GetCompareLength() const override;
  size_t GetEcPointLength() const override;

  std::string Unblind(absl::string_view input) const override;

  void SetHashType(yacl::crypto::HashAlgorithm hash_type) {
    hash_type_ = hash_type;
  }

 private:
  std::array<uint8_t, kEccKeySize> sk_inv_;
  yacl::crypto::HashAlgorithm hash_type_ = yacl::crypto::HashAlgorithm::BLAKE3;
};

}  // namespace spu::psi
