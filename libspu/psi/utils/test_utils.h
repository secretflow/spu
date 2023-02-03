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

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "yacl/crypto/base/hash/hash_utils.h"

#include "libspu/psi/psi.pb.h"

namespace spu::psi::test {

inline std::vector<std::string> CreateRangeItems(size_t begin, size_t size) {
  std::vector<std::string> ret;
  for (size_t i = 0; i < size; i++) {
    ret.push_back(std::to_string(begin + i));
  }
  return ret;
}

inline std::vector<std::string> GetIntersection(
    const std::vector<std::string> &items_a,
    const std::vector<std::string> &items_b) {
  std::set<std::string> set(items_a.begin(), items_a.end());
  std::vector<std::string> ret;
  for (const auto &s : items_b) {
    if (set.count(s) != 0) {
      ret.push_back(s);
    }
  }
  return ret;
}

inline std::vector<uint128_t> CreateItemHashes(size_t begin, size_t size) {
  std::vector<uint128_t> ret;
  for (size_t i = 0; i < size; i++) {
    ret.push_back(yacl::crypto::Blake3_128(std::to_string(begin + i)));
  }
  return ret;
}

inline std::optional<CurveType> GetOverrideCurveType() {
  if (const auto *env = std::getenv("OVERRIDE_CURVE")) {
    if (std::strcmp(env, "25519") == 0) {
      return CurveType::CURVE_25519;
    }
    if (std::strcmp(env, "FOURQ") == 0) {
      return CurveType::CURVE_FOURQ;
    }
  }
  return {};
}

}  // namespace spu::psi::test
