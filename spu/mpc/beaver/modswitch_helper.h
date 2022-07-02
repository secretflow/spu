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

#include <mutex>
#include <optional>

#include "absl/types/span.h"
#include "seal/context.h"

#include "spu/core/array_ref.h"

namespace spu::mpc {

// Helper to modulus switch between [0, Q) <-> [0, t)
class ModulusSwitchHelper {
 public:
  explicit ModulusSwitchHelper(const seal::SEALContext &seal_context,
                               uint32_t base_mod_bitlen);

  // Given x \in [0, t), compute round(Q/t*x) \in [0, Q)
  // Let Q = k*t + r for k := floor(Q/t), r := Q mod t
  // round(Q/t*x) = k*x + round(r*x/t)
  // round(Q/t*x) mod qi = ((k mod qi)*x + round(r*x/t)) mod qi
  void ModulusUpAt(absl::Span<const uint32_t> src, size_t mod_idx,
                   absl::Span<uint64_t> out) const;
  void ModulusUpAt(absl::Span<const uint64_t> src, size_t mod_idx,
                   absl::Span<uint64_t> out) const;

  // Cast [-2^{k-1}, 2^{k-1}) to [0, qj)
  void CenteralizeAt(absl::Span<const uint32_t> src, size_t mod_idx,
                     absl::Span<uint64_t> out) const;
  void CenteralizeAt(absl::Span<const uint64_t> src, size_t mod_idx,
                     absl::Span<uint64_t> out) const;

  // out = round(src*t/Q)
  void ModulusDownRNS(absl::Span<const uint64_t> src,
                      absl::Span<uint32_t> out) const;
  void ModulusDownRNS(absl::Span<const uint64_t> src,
                      absl::Span<uint64_t> out) const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_{nullptr};
};

}  // namespace spu::mpc
