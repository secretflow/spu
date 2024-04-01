// Copyright 2024 Ant Group Co., Ltd.
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
#include <memory>

#include "absl/types/span.h"
#include "seal/util/ntt.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

namespace spu::mpc::cheetah {

// Compute x * y mod p for encrypted x and plaintext y
class SIMDMulProt : public EnableCPRNG {
 public:
  static constexpr int kNoiseFloodBits = 40;

  explicit SIMDMulProt(uint64_t simd_lane, uint64_t prime_modulus);

  ~SIMDMulProt();

  SIMDMulProt& operator=(const SIMDMulProt&) = delete;

  SIMDMulProt(const SIMDMulProt&) = delete;

  SIMDMulProt(SIMDMulProt&&) = delete;

  void EncodeBatch(absl::Span<const uint64_t> array,
                   absl::Span<RLWEPt> batch_out) const;

  void EncodeSingle(absl::Span<const uint64_t> array, RLWEPt& out) const;

  void DecodeSingle(const RLWEPt& poly, absl::Span<uint64_t> array) const;

  void SymEncrypt(absl::Span<const RLWEPt> polys,
                  const RLWESecretKey& secret_key,
                  const seal::SEALContext& context, bool save_seed,
                  absl::Span<RLWECt> out) const;

  void MulThenReshareInplace(absl::Span<RLWECt> ct, absl::Span<const RLWEPt> pt,
                             absl::Span<const uint64_t> share_mask,
                             const RLWEPublicKey& public_key,
                             const seal::SEALContext& context);

  void MulThenReshareInplaceOneBit(absl::Span<RLWECt> ct,
                                   absl::Span<const RLWEPt> pt,
                                   absl::Span<uint64_t> share_mask,
                                   const RLWEPublicKey& public_key,
                                   const seal::SEALContext& context);

  inline int64_t SIMDLane() const { return simd_lane_; }

  const seal::Modulus& modulus() const { return prime_modulus_; }

 private:
  void NoiseFloodInplace(RLWECt& ct, const seal::SEALContext& context);

  int64_t simd_lane_;
  seal::Modulus prime_modulus_;
  std::unique_ptr<seal::util::NTTTables> encode_tabl_;
};

}  // namespace spu::mpc::cheetah
