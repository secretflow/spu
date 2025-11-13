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

#include "libspu/kernel/hal/permute.h"

namespace spu::kernel::hlo {

/// Some conventions about permutation semantic:
///   1. The `perm` in MPC layer is so-called shared with "composition"
///   semantic, i.e. let \pi is a perm, then P0 holds \pi0, P1 holds \pi1, and
///   \pi = \pi1 o \pi0 (`o` means composition of two permutations).
///   2. There exists another semantic of `perm` (We call it "Additive"
///   semantic). The `perm` is a vector of destinations, and each element of the
///   vector is secret shared across the parties. e.g. let \pi = \pi0 + \pi1 if
///   we have 2 Parties. We implement the ops under this semantic of `perm` in
///   HAL layer based on the "composition" semantic of perm.
///   (REF: https://eprint.iacr.org/2019/695.pdf)
///
///   Note: In HLO layer, we always assume the perm is "Additive" semantic.

// Inverse permute vector `inputs` over permutation `perm`
// Let [n] = {0,1,2,...,n-1}, then perm: [n] -> [n] should be an invertible
// permutation, we denote prem^{-1} as its inversion.
// For each input x, we get y = perm^{-1} (x), i.e. y[i] = x[perm^{-1}(i)]
//
// Note: to simplify the implementation, we FORCE the visibility of inputs to be
// the SAME (for Private, the OWNER should also be the SAME).
// IMPORTANT NOTE: when perm is Private (owner i), and inputs include some mix
// of either Secret or Private (with owner j != i), you should Seal the Private
// inputs (with owner j != i) first, and do permute once to improve performance.
std::vector<spu::Value> InvPermute(SPUContext* ctx,
                                   absl::Span<const spu::Value> inputs,
                                   const spu::Value& perm, int64_t perm_dim);

// Permute vector `inputs` over permutation `perm`
// For each input x, we get y = perm(x), i.e. y[i] = x[perm(i)]
// Note: to simplify the implementation, we force the visibility of inputs to be
// the same (for Private, the owner should also be the same).
//
// Note: to simplify the implementation, we FORCE the visibility of inputs to be
// the SAME (for Private, the OWNER should also be the SAME).
// IMPORTANT NOTE: when perm is Private (owner i), and inputs include some mix
// of either Secret or Private (with owner j != i), you should Seal the Private
// inputs (with owner j != i) first, and do permute once to improve performance.
std::vector<spu::Value> Permute(SPUContext* ctx,
                                absl::Span<const spu::Value> inputs,
                                const spu::Value& perm, int64_t perm_dim);

std::vector<spu::Value> GeneralPermute(SPUContext* ctx,
                                       absl::Span<const spu::Value> inputs,
                                       const spu::Value& perm,
                                       int64_t perm_dim);
}  // namespace spu::kernel::hlo
