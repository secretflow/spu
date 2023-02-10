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

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/base/int128.h"
#include "yacl/crypto/base/hash/hash_utils.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/cuckoo_index.h"

namespace spu::psi {

// GeneralizedCuckooHash options
// now support (2,2), (3,2) gch
// Reference:
// DW07. M. Dietzfelbinger and C. Weidling.
// Balanced allocation and dictionaries with tightly packed constant size bins
//
CuckooIndex::Options GetCuckooHashOption(size_t bin_size, size_t hash_num,
                                         size_t items_size);

// abstract interface for psi hash table
class IPsiHashTable {
 public:
  virtual ~IPsiHashTable() {}

  virtual void Insert(absl::Span<const std::string> items) = 0;
};

class GeneralizedCuckooHashTable : public IPsiHashTable {
 public:
  explicit GeneralizedCuckooHashTable(CuckooIndex::Options options,
                                      size_t bin_data_num, uint128_t seed = 0);

  void Insert(yacl::ByteContainerView item_data, size_t input_offset);
  void Insert(yacl::ByteContainerView item);
  void Insert(absl::Span<const std::string> items) override;

  const std::vector<std::vector<CuckooIndex::Bin>> &bins() const {
    return bins_;
  }

  const std::vector<uint64_t> &GetItemsHashLow64() const {
    return items_hash_low64_;
  }

  // Returns the current fill rate of the hash table and stash.
  inline double FillRate() const noexcept {
    return static_cast<double>(inserted_items_) /
           (static_cast<double>(gch_options_.NumBins()) * max_items_per_bin_);
  }

  const CuckooIndex::Options &GetCuckooOptions() const { return gch_options_; }

  size_t GetMaxItemsPerBin() const { return max_items_per_bin_; }

 protected:
  CuckooIndex::Options gch_options_;
  // max data number per bin
  size_t max_items_per_bin_;
  uint128_t seed_;
  std::vector<std::vector<CuckooIndex::Bin>> bins_;
  std::vector<std::vector<uint64_t>> hashes_;
  std::vector<uint64_t> items_hash_low64_;
  size_t inserted_items_ = 0;

  // Randomness source for location function sampling.
  std::mt19937_64 gen_;

  std::uniform_int_distribution<std::uint32_t> uniform_hash_idx_;
  std::uniform_int_distribution<std::uint32_t> uniform_data_idx_;
};

class SimpleHashTable : public IPsiHashTable {
 public:
  explicit SimpleHashTable(CuckooIndex::Options options, uint128_t seed = 0);

  void Insert(yacl::ByteContainerView item_data,
              const std::vector<uint64_t> &hash_bin_idx);
  void Insert(yacl::ByteContainerView item);
  void Insert(absl::Span<const std::string> items) override;

  const std::vector<std::vector<CuckooIndex::Bin>> &bins() const {
    return bins_;
  }

  const std::vector<uint64_t> &GetItemsHashLow64() const {
    return items_hash_low64_;
  }

  const CuckooIndex::Options &GetCuckooOptions() const { return gch_options_; }

  const std::vector<size_t> &GetConflictIdx() const { return conflict_idx_; }

 protected:
  CuckooIndex::Options gch_options_;
  uint128_t seed_;
  std::vector<std::vector<CuckooIndex::Bin>> bins_;
  std::vector<uint64_t> items_hash_low64_;
  size_t inserted_items_ = 0;
  std::vector<size_t> conflict_idx_;
};

}  // namespace spu::psi
