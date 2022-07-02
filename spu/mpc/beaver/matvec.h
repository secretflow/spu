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

#include "absl/types/span.h"
#include "yasl/base/buffer.h"

#include "spu/core/array_ref.h"
#include "spu/mpc/beaver/matvec_helper.h"

namespace seal {
class GaloisKeys;
class SEALContext;
class Ciphertext;
class Encryptor;
}  // namespace seal

namespace spu::mpc {
class ModulusSwitchHelper;

// HE-based Protocol for computing MatVec from encrypted vector and plain
// matrix.
// Requirements:
//  - min(nrows, ncols) <= num_slots / 2
//  - max(nrows, ncols) <= num_slots
class MatVecProtocol {
 protected:
  struct Impl;
  std::shared_ptr<Impl> impl_;  // private implementation

 public:
  explicit MatVecProtocol(const seal::GaloisKeys &rot_keys,
                          const seal::SEALContext &context);

  size_t num_slots() const;

  // Encrypt a compact vector and serialize it to buffer.
  yasl::Buffer EncryptVector(ArrayRef vec,
                             const MatVecHelper::MatViewMeta &meta,
                             const seal::Encryptor &sym_encryptor) const;

  // Encode a compact submatrix. The submatrix is defined by `meta`.
  void EncodeSubMatrix(ArrayRef mat, const MatVecHelper::MatViewMeta &meta,
                       std::vector<seal::Plaintext> *out) const;

  void EncodeSubMatrix(ArrayRef mat, const MatVecHelper::MatViewMeta &meta,
                       absl::Span<seal::Plaintext> out) const;

  void EncodeSubMatrix(ArrayRef mat, const MatVecHelper::MatViewMeta &meta,
                       const ModulusSwitchHelper &ms_helper,
                       size_t target_prime_index,
                       absl::Span<seal::Plaintext> out) const;

  void Compute(const seal::Ciphertext &enc_vec,
               absl::Span<const seal::Plaintext> ecd_mat,
               const MatVecHelper::MatViewMeta &meta,
               seal::Ciphertext *out) const;
};

}  // namespace spu::mpc
