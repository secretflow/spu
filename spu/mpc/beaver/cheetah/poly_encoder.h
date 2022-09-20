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

#include "spu/core/array_ref.h"
#include "spu/mpc/beaver/cheetah/modswitch_helper.h"
#include "spu/mpc/beaver/cheetah/types.h"

namespace spu::mpc {

class PolyEncoder {
 public:
  explicit PolyEncoder(const seal::SEALContext &context,
                       ModulusSwitchHelper ms_helper);
  // clang-format off
  // Math:
  //    Forward(x) * Backward(y) mod X^N + 1 gives the inner product <x, y>
  // When doing in encryption, we need the encrypted part to be scaled up by some fixed `Delta` 
  // For example Enc(Forward(Delta*x)) * Backward(y) can give the <x,  y> without error.
  // Or we can encrypt the backward part, i.e., Enc(Backward(Delta*y)) * Forward(x) also gives the <x, y> without error.
  // clang-format on
  void Forward(const ArrayRef &vec, RLWEPt *out, bool scaleup = true) const;

  void Backward(const ArrayRef &vec, RLWEPt *out, bool scaleup = false) const;

  const ModulusSwitchHelper &ms_helper() const { return ms_helper_; }

 private:
  size_t poly_deg_{0};
  // Take the copy
  ModulusSwitchHelper ms_helper_;
};

}  // namespace spu::mpc
