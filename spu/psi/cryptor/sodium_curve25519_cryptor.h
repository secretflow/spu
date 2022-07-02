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

#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "yasl/base/exception.h"
#include "yasl/link/link.h"

#include "spu/psi/cryptor/ecc_cryptor.h"

namespace spu {

//
// avx2 asm code reference:
// https://eprint.iacr.org/2015/943.pdf
// Sandy2x: New Curve25519 Speed Records
// Table1 Performance results compare with “floodyberry” [7].
//
// fix Side Channel Attack in
// https://eprint.iacr.org/2017/806
// May the Fourth Be With You: A Microarchitectural Side Channel Attack on
// Several Real-World Applications of Curve25519
// The vulnerability has been assigned CVE-2017-0379
//
class SodiumCurve25519Cryptor : public IEccCryptor {
 public:
  SodiumCurve25519Cryptor() {
    //
    // curve25519 secret keys range:
    //   2^254 + 8*{0,1,2, 2^251-1}
    // convert 32byte randombytes to curve25519 secret key range
    //
    private_key_[0] &= 248;
    private_key_[31] &= 127;
    private_key_[31] |= 64;
  }

  ~SodiumCurve25519Cryptor() override = default;

  CurveType GetCurveType() const override { return CurveType::Curve25519; }

  void EccMask(absl::Span<const char> batch_points,
               absl::Span<char> dest_points) const override;

  std::vector<uint8_t> KeyExchange(
      const std::shared_ptr<yasl::link::Context> &link_ctx);
};

}  // namespace spu