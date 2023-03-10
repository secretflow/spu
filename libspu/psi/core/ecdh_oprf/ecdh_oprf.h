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

#include "absl/strings/escaping.h"
#include "absl/types/span.h"
#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "spdlog/spdlog.h"
#include "yacl/base/byte_container_view.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/cryptor/ecc_cryptor.h"

namespace spu::psi {

enum class OprfType {
  Basic,
  // ToDo add this type support
  // RfcVOprf,
};

// reference:
// voprf rfc draft:
// https://datatracker.ietf.org/doc/draft-irtf-cfrg-voprf/ 3.3 Context API
//
// [TCRSTW21] "A Fast and Simple Partially Oblivious PRF, with Applications",
// <https://eprint.iacr.org/2021/864>.

//
// Client(input)                                   Server(skS)
// ----------------------------------------------------------------------
//  blindedElement = Blind(input)
//                       blindedElement
//                        ---------->
//
//                      evaluatedElement = Evaluate(skS, blindedElement)
//
//                       evaluatedElement
//                        <----------
//
// output = Finalize(input, evaluatedElement)
//

class IEcdhOprf {
 public:
  IEcdhOprf() {
    SPU_ENFORCE(RAND_bytes(&private_key_[0], kEccKeySize) == 1,
                "Cannot create random private key");
  }

  virtual ~IEcdhOprf() { OPENSSL_cleanse(&private_key_[0], kEccKeySize); }

  virtual OprfType GetOprfType() const = 0;

  virtual size_t GetCompareLength() const {
    if (compare_length_) {
      return compare_length_;
    }
    return kEccKeySize;
  }

  virtual size_t GetEcPointLength() const { return kEccKeySize; }

  void SetCompareLength(size_t compare_length) {
    SPU_ENFORCE(compare_length <= kEccKeySize);
    compare_length_ = compare_length;
  }

  void SetPrivateKey(yacl::ByteContainerView private_key) {
    SPU_ENFORCE(private_key.size() == kEccKeySize);

    std::memcpy(private_key_, private_key.data(), private_key.size());
  }

 protected:
  uint8_t private_key_[kEccKeySize];
  size_t compare_length_ = 0;
};

class IEcdhOprfServer : public IEcdhOprf {
 public:
  IEcdhOprfServer() = default;
  // set private_key
  explicit IEcdhOprfServer(yacl::ByteContainerView private_key) {
    SetPrivateKey(private_key);
  }

  ~IEcdhOprfServer() override = default;

  /**
   * @brief Evaluate takes serialized representations of blinded group elements
   * from the client as inputs
   *
   * @param blinded_element  blinded data masked by client's temp private key
   * @return std::string mask blinded data with server's private key
   */
  virtual std::string Evaluate(absl::string_view blinded_element) const = 0;

  virtual std::vector<std::string> Evaluate(
      absl::Span<const std::string> blinded_element) const;

  /**
   * @brief FullEvaluate takes input values, and it is useful for applications
   * that need to compute the whole OPRF protocol on the server side only.
   *
   * @param input   server's input data
   * @return std::string   H2(x,H1(x)^sk)
   */
  virtual std::string FullEvaluate(yacl::ByteContainerView input) const = 0;

  /**
   * @brief SimpleEvaluate takes input values, and it is useful for applications
   * that need to compute the whole OPRF protocol on the server side only.
   *
   * @param input   server's input data
   * @return std::string   H2(H1(x)^sk)
   */
  virtual std::string SimpleEvaluate(yacl::ByteContainerView input) const = 0;

  virtual std::vector<std::string> FullEvaluate(
      absl::Span<const std::string> input) const;

  virtual std::array<uint8_t, kEccKeySize> GetPrivateKey() const {
    std::array<uint8_t, kEccKeySize> key_array{};
    std::memcpy(key_array.data(), &private_key_[0], kEccKeySize);
    return key_array;
  }
};

class IEcdhOprfClient : public IEcdhOprf {
 public:
  IEcdhOprfClient() = default;

  ~IEcdhOprfClient() override = default;

  //
  /**
   * @brief Blind the input, use client temp private key
   *
   * @param input   client input data
   * @return std::string   blinded data
   */
  virtual std::string Blind(absl::string_view input) const = 0;

  virtual std::vector<std::string> Blind(
      absl::Span<const std::string> input) const;

  virtual std::string Unblind(absl::string_view input) const = 0;

  /**
   * @brief unblind evaluated_element, and hash
   *
   * @param item   client input data
   * @param evaluated_element
   * @return std::string masked data with server's private key
   */
  virtual std::string Finalize(absl::string_view item,
                               absl::string_view evaluated_element) const = 0;

  virtual std::string Finalize(absl::string_view evaluated_element) const = 0;

  /**
   * @brief unblind evaluated_element, and hash
   *
   * @param item   client input data
   * @param evaluated_element
   * @return std::string masked data with server's private key
   */
  virtual std::vector<std::string> Finalize(
      absl::Span<const std::string> items,
      absl::Span<const std::string> evaluated_element) const;

  virtual std::vector<std::string> Finalize(
      absl::Span<const std::string> evaluated_element) const;
};

}  // namespace spu::psi
