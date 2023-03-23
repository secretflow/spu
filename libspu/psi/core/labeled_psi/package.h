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

#include <vector>

#include "apsi/crypto_context.h"
#include "apsi/seal_object.h"
#include "gsl/span"
#include "seal/seal.h"

namespace spu::psi {

struct PlainResultPackage {
  std::uint32_t bundle_idx;

  std::vector<std::uint64_t> psi_result;

  std::uint32_t label_byte_count;

  std::uint32_t nonce_byte_count;

  std::vector<std::vector<std::uint64_t>> label_result;
};

class ResultPackage {
 public:
  PlainResultPackage extract(const apsi::CryptoContext& crypto_context);

  std::uint32_t bundle_idx;

  seal::compr_mode_type compr_mode = seal::Serialization::compr_mode_default;

  apsi::SEALObject<seal::Ciphertext> psi_result;

  std::uint32_t label_byte_count;

  std::uint32_t nonce_byte_count;

  std::vector<apsi::SEALObject<seal::Ciphertext>> label_result;
};  // struct ResultPackage

}  // namespace spu::psi
