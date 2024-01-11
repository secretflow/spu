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
#include "absl/types/span.h"

#include "libspu/mpc/cheetah/rlwe/types.h"

namespace spu::mpc::cheetah {

// REF: Efficient Homomorphic Conversion Between (Ring) LWE Ciphertexts
// https://eprint.iacr.org/2020/015.pdf
void GenerateGaloisKeyForPacking(const seal::SEALContext &context,
                                 const RLWESecretKey &key, bool save_seed,
                                 GaloisKeys *out);

// REF: BumbleBee: Secure Two-party Inference Framework for Large Transformers
// https://eprint.iacr.org/2023/1678
class PackingHelper {
 public:
  PackingHelper(size_t gap, const seal::GaloisKeys &galois_keys,
                const seal::SEALContext &gk_context,
                const seal::SEALContext &context);

  // require ct_array.size() == gap
  void PackingWithModulusDrop(absl::Span<RLWECt> rlwes, RLWECt &packed) const;

 private:
  void MultiplyFixedScalarInplace(RLWECt &ct) const;

  void doPackingRLWEs(absl::Span<RLWECt> rlwes, RLWECt &out) const;

  size_t gap_;
  const seal::GaloisKeys &galois_keys_;
  const seal::SEALContext &gk_context_;
  const seal::SEALContext &context_;

  std::vector<seal::util::MultiplyUIntModOperand> inv_gap_;
};

// lwes[0, N) -> RLWE[0], lwe[N, 2N) -> RLWE[1] ....
// Return the number of output RLWE ciphertexts.
size_t PackLWEs(absl::Span<const LWECt> lwes, const GaloisKeys &galois,
                const seal::SEALContext &context, absl::Span<RLWECt> rlwes);

size_t PackLWEs(absl::Span<const PhantomLWECt> lwes, const GaloisKeys &galois,
                const seal::SEALContext &context, absl::Span<RLWECt> rlwes);
}  // namespace spu::mpc::cheetah