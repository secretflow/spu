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

#include "libspu/core/memref.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols;

// clang-format off
// REF: CrypTFlow2: Practical 2-party secure inference.
// [1{x > y}]_B <- CMP(x, y) for two private input
//
// Math:
//   1. break into digits:
//       x = x0 || x1 || ..., ||xd
//       y = y0 || y1 || ..., ||yd
//      where 0 <= xi,yi < 2^{radix}
//   2. Use 1-of-2^{radix} OTs to compute lt_i = [1{xi < yi}]_B and eq_i = [1{xi = yi}]_B
//   3. Recursively apply 1{x < y} = 1{xd < xd} ^ (1{xd = xd} & 1{x[0:d) < y[0:d)})
//     - Tree-based reduction
// There is a trade-off between round and communication.
// A larger radix renders a smaller number of rounds but a larger  communication.
// clang-format on
class CompareProtocol {
 public:
  // REQUIRE 1 <= compare_radix <= 4
  explicit CompareProtocol(const std::shared_ptr<BasicOTProtocols>& base,
                           size_t compare_radix = 4);

  ~CompareProtocol();

  MemRef Compute(const MemRef& inp, bool greater_than, int64_t bitwidth = 0);

  std::array<MemRef, 2> ComputeWithEq(const MemRef& inp, bool greater_than,
                                      int64_t bitwidth = 0);

 private:
  MemRef DoCompute(const MemRef& inp, bool greater_than, MemRef* eq = nullptr,
                   int64_t bitwidth = 0);

  MemRef TraversalAND(MemRef cmp, MemRef eq, size_t num_input,
                      size_t num_digits);

  // Require num_digits to be two-power value
  MemRef TraversalANDFullBinaryTree(MemRef cmp, MemRef eq, size_t num_input,
                                    size_t num_digits);
  std::array<MemRef, 2> TraversalANDWithEq(MemRef cmp, MemRef eq,
                                           size_t num_input, size_t num_digits);

  // Require num_digits to be two-power value
  std::array<MemRef, 2> TraversalANDWithEqFullBinaryTree(MemRef cmp, MemRef eq,
                                                         size_t num_input,
                                                         size_t num_digits);
  size_t compare_radix_;
  bool is_sender_{false};
  std::shared_ptr<BasicOTProtocols> basic_ot_prot_;
};

}  // namespace spu::mpc::cheetah
