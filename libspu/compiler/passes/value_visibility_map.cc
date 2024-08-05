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

#include "libspu/compiler/passes/value_visibility_map.h"

#include "libspu/core/prelude.h"

namespace mlir::pphlo {
namespace {

Visibility ComputePromotedVisibility(Visibility v1, Visibility v2) {
  if (v1 == v2) {
    return v1;
  }
  if (v1 == Visibility::VIS_SECRET || v2 == Visibility::VIS_SECRET) {
    return Visibility::VIS_SECRET;
  }
  return Visibility::VIS_PUBLIC;
}

} // namespace

Visibility ValueVisibilityMap::getValueVisibility(const Value &v) const {
  const auto &iter = storage.find(v);
  SPU_ENFORCE(iter != storage.end());
  return iter->second;
}

void ValueVisibilityMap::setValueVisibility(const Value &val, Visibility vis) {
  const auto &iter = storage.find(val);
  if (iter != storage.end()) {
    // Merge
    storage[val] = ComputePromotedVisibility(iter->second, vis);
  } else {
    storage[val] = vis;
  }
}

} // namespace mlir::pphlo
