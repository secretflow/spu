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

#include "libspu/mpc/kernel.h"

namespace spu::mpc::semi2k {

// ref: Ciphers for MPC and FHE
// https://eprint.iacr.org/2016/687.pdf
//
// LowMC cipher is a MPC-friendly block cipher which minimizes the depth and
// numbers of And Gates.
// For current implementation, we only support 128-bit key security. But user
// can change the data complexity to achieve higher efficiency.
//
// NOTE: Although LowMC is protocol agnostic (only depends on some boolean ops),
// but we still implement it in each protocol kernel now, for efficiency
// consideration.
class LowMcB : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "lowmc_b"; }

  // the concrete cost depends on the data complexity
  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;

  // inner function, mark as public only for testing
  NdArrayRef encrypt(KernelEvalContext* ctx, const NdArrayRef& in,
                     uint128_t key, uint128_t seed  // single key now
  ) const;
};

// For multi-key condition, we use the scheme in:
// REF: https://eprint.iacr.org/2019/518
//
// If we have m keys, each key has k bits, logically:
//   1. Concat all these keys and get mk-bits single key `X`.
//   2. Each party sample the same random binary matrix `M` with shape (mk, n),
//   where n is the bits that SPU can handle (e.g. 128).
//   3. Then we compute `Y = gf2dot(X, M)`, and use `Y` as the input for LowMc
//   encryption.
//
// Collision Prob p: about 2^{-n+q}, where q ~= 2 * log2(D), D is the total
// number of encoding.
// i.e. when n = 128, D = 2**20 (1M) , p ~= 2^{-88}
//      when n = 128, D = 2**30 (1B) , p ~= 2^{-68}
class MultiKeyLowMcB : public MultiKeyLowMcKernel {
 public:
  static constexpr const char* kBindName() { return "multi_key_lowmc_b"; }

  // the concrete cost depends on the data complexity
  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx,
                  const std::vector<NdArrayRef>& inputs) const override;
};

}  // namespace spu::mpc::semi2k
