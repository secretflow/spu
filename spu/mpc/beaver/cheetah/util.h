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
#include <set>

#include "yasl/base/buffer.h"

#include "spu/mpc/beaver/cheetah/types.h"

namespace spu {
class ArrayRef;
}

namespace spu::mpc {

// requires ciphertext.is_ntt_form() is `false`
//          ciphertext.size() is `2`
void RemoveCoefficientsInplace(RLWECt& ciphertext,
                               const std::set<size_t>& to_remove);

void KeepCoefficientsInplace(RLWECt& ciphertext,
                             const std::set<size_t>& to_keep);

// Erase the memory automatically
struct AutoMemGuard {
  AutoMemGuard(ArrayRef* obj);

  AutoMemGuard(RLWEPt* pt);

  ~AutoMemGuard();

  ArrayRef* obj_{nullptr};
  RLWEPt* pt_{nullptr};
};

}  // namespace spu::mpc
