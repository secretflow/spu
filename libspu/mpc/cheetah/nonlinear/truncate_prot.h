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

#include <memory>

#include "libspu/core/array_ref.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols;

// Implementation the one-bit approximate truncation
// Ref: Huang et al. "Cheetah: Lean and Fast Secure Two-Party Deep Neural
// Network Inference"
//  https://eprint.iacr.org/2022/207.pdf
//
// [(x >> s) + e]_A <- Truncate([x]_A, s) with |e| <= 1 probabilistic error
//
// Math:
//   Given x = x0 + x1 mod 2^k
//   x >> s \approx (x0 >> s) + (x1 >> s) - w * 2^{k - s} mod 2^k
//   where w = 1{x0 + x1 > 2^{k} - 1} indicates whether the sum wrap round 2^k
class TruncateProtocol {
 public:
  enum class MSB_st {
    zero,
    one,
    unknown,
  };

  struct Meta {
    MSB_st msb;
    bool use_heuristic;  // not implemented yet
    bool signed_arith;
    size_t shift_bits;

    Meta()
        : msb(MSB_st::unknown),
          use_heuristic(false),
          signed_arith(true),
          shift_bits(0) {}
  };

  explicit TruncateProtocol(std::shared_ptr<BasicOTProtocols> base);

  ~TruncateProtocol();

  ArrayRef Compute(const ArrayRef &inp, Meta meta);

 private:
  ArrayRef ComputeWrap(const ArrayRef &inp, const Meta &meta);

  // w = msbA | msbB
  ArrayRef MSB0ToWrap(const ArrayRef &inp, size_t shift_bits);

  // w = msbA & msbB
  ArrayRef MSB1ToWrap(const ArrayRef &inp, size_t shift_bits);

  std::shared_ptr<BasicOTProtocols> basic_ot_prot_{nullptr};
};

}  // namespace spu::mpc::cheetah
