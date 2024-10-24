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

#include "openssl/crypto.h"
#include "openssl/rand.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/cryptor/ecc_cryptor.h"

namespace spu::psi {

class FourQEccCryptor : public IEccCryptor {
 public:
  FourQEccCryptor() = default;

  ~FourQEccCryptor() override = default;

  CurveType GetCurveType() const override { return CurveType::CURVE_FOURQ; }

  void EccMask(absl::Span<const char> batch_points,
               absl::Span<char> dest_points) const override;

  std::vector<uint8_t> HashToCurve(absl::Span<const char> input) const override;
};

}  // namespace spu::psi