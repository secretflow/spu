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

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols;

// Given [x] the share in modulo p, to compute the share in
// modulo 2^k such that 2^k > p.
// Assume |x| < p/4 to apply the positive heuristic.
//
// Also, we can handle the 1-bit approximated truncation inside
// this prime-to-ring conversion.
// Math:
// Given a prime share h0 + h1 = h mod p,
// to convert it to s0 + s1 = h mod 2^k.
// Basically we compute the wrap bit w = 1{h0 + h1 >= p}.
// Then s0 = h0 - p * w mod 2^k and s1 = h1 - p * w mod 2^k.
// The wrap bit w is computed as mod-2^k share, i.e., w0 + w1 = w mod 2^k.
//
// For the 1-bit approximated truncation,
// we further compute t0 + t1 = h/2^d + e mod 2^k for some arithmetic right
// shift by d-unit and 1-bit error |e| <= 1.
//
// That is t0 = (h0/2^d) - (p/2^d) * w0 mod 2^k,
//         t1 = (h1/2^d) - (p/2^d) * w1 mod 2^k.
class PrimeRingExtendProtocol {
 public:
  static constexpr size_t kHeuristicBound = 2;

  struct Meta {
    uint64_t prime;
    FieldType dst_ring;
    int64_t dst_width;
    // To do (approximated) truncation if specified.
    std::optional<int> truncate_nbits;
  };

  explicit PrimeRingExtendProtocol(
      const std::shared_ptr<BasicOTProtocols> &base);

  ~PrimeRingExtendProtocol() = default;

  NdArrayRef Compute(const NdArrayRef &inp, const Meta &meta);

 private:
  NdArrayRef MSB0ToWrap(const NdArrayRef &inp, const Meta &meta);

  std::shared_ptr<BasicOTProtocols> basic_ot_prot_ = nullptr;
};

}  // namespace spu::mpc::cheetah
