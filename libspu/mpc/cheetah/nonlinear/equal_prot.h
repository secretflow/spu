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

// REF: CrypTFlow2: Practical 2-party secure inference.
// [1{x = y}]_B <- EQ(x, y) for two private input
//
// Math:
//   1. break into digits:
//       x = x0 || x1 || ..., ||xd
//       y = y0 || y1 || ..., ||yd
//      where 0 <= xi,yi < 2^{radix}
//   2. Use 1-of-2^{radix} OTs to compute the bit eq_i = [1{xi = yi}]_B
//   3. Tree-based AND 1{x = y} = AND_i eq_i
// There is a trade-off between round and communication.
// A larger radix renders a smaller number of rounds but a larger
// communication.
class EqualProtocol {
 public:
  // REQUIRE 1 <= compare_radix <= 8.
  explicit EqualProtocol(std::shared_ptr<BasicOTProtocols> base,
                         size_t compare_radix = 4);

  ~EqualProtocol();

  ArrayRef Compute(const ArrayRef& inp);

 private:
  size_t compare_radix_;
  bool is_sender_{false};
  std::shared_ptr<BasicOTProtocols> basic_ot_prot_;
};

}  // namespace spu::mpc::cheetah
