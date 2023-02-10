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

// Author (Wen-jie Lu)
#pragma once

#include <mutex>
#include <optional>

#include "absl/types/span.h"
#include "seal/context.h"

#include "libspu/core/array_ref.h"

namespace spu::mpc::cheetah {

// Helper to modulus switch between [0, Q) <-> [0, 2^k) for k <= 128.
// Q is defined in the key_context_data() from SEALContext
class ModulusSwitchHelper {
 public:
  explicit ModulusSwitchHelper(const seal::SEALContext& seal_context,
                               uint32_t base_mod_bitlen);

  seal::parms_id_type parms_id() const;

  // return `k` in 2^k
  uint32_t base_mod_bitlen() const;

  // return number of modulus such that Q = \prod_i q_i.
  uint32_t coeff_modulus_size() const;

  // Given x \in [0, 2^k), compute round(Q/2^k*x) \in [0, Q)
  void ModulusUpAt(const ArrayRef& src, size_t mod_idx,
                   absl::Span<uint64_t> out) const;

  // Cast [-2^{k-1}, 2^{k-1}) to [0, qj)
  void CenteralizeAt(const ArrayRef& src, size_t mod_idx,
                     absl::Span<uint64_t> out) const;

  // Given x' in \ [0, Q), compute round(x'*2^k/Q) \in [0, 2^k)
  ArrayRef ModulusDownRNS(FieldType field,
                          absl::Span<const uint64_t> src) const;

  void ModulusDownRNS(absl::Span<const uint64_t> src, ArrayRef out) const;

  void ModulusDownRNS(absl::Span<const uint64_t> src,
                      absl::Span<uint32_t> out) const;
  void ModulusDownRNS(absl::Span<const uint64_t> src,
                      absl::Span<uint64_t> out) const;
  void ModulusDownRNS(absl::Span<const uint64_t> src,
                      absl::Span<uint128_t> out) const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_{nullptr};
};

}  // namespace spu::mpc::cheetah
