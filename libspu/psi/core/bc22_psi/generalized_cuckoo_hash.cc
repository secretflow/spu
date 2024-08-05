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

#include "libspu/psi/core/bc22_psi/generalized_cuckoo_hash.h"

#include <set>
#include <utility>

#include "absl/strings/escaping.h"
#include "spdlog/spdlog.h"
#include "yacl/base/int128.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/parallel.h"

namespace spu::psi {

namespace {

constexpr size_t kDefaultHashNum = 2;

// Permutation-Based Hashing
// [PSSZ15] In USENIX Security 2015
// Phasing: Private set intersection using permutation-based hashing
// [BC22] Section 2.4
// h0(x) = hash(x),
// h1(x) = h0(x) xor fingerprint(x).
std::vector<uint64_t> GetBinIdx(const CuckooIndex::Options &options,
                                uint128_t item_hash, uint64_t item_hash_u64) {
  SPU_ENFORCE(options.num_hash == kDefaultHashNum);
  size_t num_bins = options.NumBins();

  CuckooIndex::HashRoom hash_room(item_hash);

  size_t bin_idx = hash_room.GetHash(0);

  std::vector<uint64_t> hash_bin_idx(options.num_hash);

  hash_bin_idx[0] = bin_idx % num_bins;
  hash_bin_idx[1] = (bin_idx ^ item_hash_u64) % num_bins;

  return hash_bin_idx;
}

}  // namespace

CuckooIndex::Options GetCuckooHashOption(size_t bin_size, size_t hash_num,
                                         size_t items_size) {
  CuckooIndex::Options options;

  options.num_input = items_size;
  options.num_stash = 0;
  options.num_hash = hash_num;

  SPU_ENFORCE(hash_num == kDefaultHashNum, "just support 2 hash");

  if (hash_num == kDefaultHashNum) {
    if (bin_size == 2) {
      options.scale_factor = 1;
    } else if (bin_size == 3) {
      options.scale_factor = 0.6;
    } else {
      SPU_THROW("unsupported");
    }
  } else {
    SPU_THROW("unsupported");
  }
  return options;
}

GeneralizedCuckooHashTable::GeneralizedCuckooHashTable(
    CuckooIndex::Options options, size_t bin_data_num, uint128_t seed)
    : gch_options_(options),
      max_items_per_bin_(bin_data_num),
      seed_(seed),
      gen_(yacl::crypto::SecureRandU64()) {
  size_t table_size = gch_options_.NumBins();
  bins_.resize(table_size);

  uniform_hash_idx_ =
      std::uniform_int_distribution<uint32_t>(0, gch_options_.num_hash - 1);
  uniform_data_idx_ =
      std::uniform_int_distribution<uint32_t>(0, max_items_per_bin_ - 1);
}

void GeneralizedCuckooHashTable::Insert(yacl::ByteContainerView item_data,
                                        size_t input_offset) {
  CuckooIndex::Bin candidate;
  candidate.set_encoded(input_offset);

  int64_t level = gch_options_.max_try_count;
  size_t bin_idx;
  while ((level--) != 0) {
    size_t rand_hash_idx = uniform_hash_idx_(gen_);

    for (uint32_t i = 0; i < gch_options_.num_hash; i++) {
      size_t hash_idx = (rand_hash_idx + i) % gch_options_.num_hash;

      bin_idx = hashes_[candidate.InputIdx()][hash_idx];

      if (bins_[bin_idx].size() < max_items_per_bin_) {
        uint64_t next_candid =
            CuckooIndex::Bin::Encode(candidate.InputIdx(), hash_idx);
        candidate.set_encoded(next_candid);

        bins_[bin_idx].push_back(candidate);
        inserted_items_++;

        return;
      }
    }

    // random select bin_idx and idx in bin, swap candidate
    size_t rand_data_idx = uniform_data_idx_(gen_);

    rand_hash_idx = uniform_hash_idx_(gen_);

    bin_idx = hashes_[candidate.InputIdx()][rand_hash_idx];

    uint64_t next_candid =
        CuckooIndex::Bin::Encode(candidate.InputIdx(), rand_hash_idx);
    candidate.set_encoded(next_candid);

    candidate = CuckooIndex::Bin(
        bins_[bin_idx][rand_data_idx].Swap(candidate.encoded()));
  }

  SPU_THROW(
      "Error insert, level:{} insert item_data:{}", level,
      absl::BytesToHexString(absl::string_view(
          reinterpret_cast<const char *>(item_data.data()), item_data.size())));
}

void GeneralizedCuckooHashTable::Insert(yacl::ByteContainerView item) {
  uint128_t item_hash = yacl::crypto::Blake3_128(item);
  size_t input_offset = hashes_.size();

  // hash_bin_idx
  std::pair<uint64_t, uint64_t> items_hash_u64 =
      yacl::DecomposeUInt128(item_hash);
  std::vector<uint64_t> hash_bin_idx =
      GetBinIdx(gch_options_, item_hash, items_hash_u64.first);
  hashes_.push_back(hash_bin_idx);

  items_hash_low64_.push_back(items_hash_u64.second);

  Insert(item, input_offset);
}

void GeneralizedCuckooHashTable::Insert(absl::Span<const std::string> items) {
  size_t input_offset = items_hash_low64_.size();
  items_hash_low64_.resize(input_offset + items.size());
  hashes_.resize(input_offset + items.size());

  yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    for (int i = begin; i < end; ++i) {
      uint128_t item_hash = yacl::crypto::Blake3_128(items[i]);

      std::pair<uint64_t, uint64_t> items_hash_u64 =
          yacl::DecomposeUInt128(item_hash);

      items_hash_low64_[input_offset + i] = items_hash_u64.second;
      hashes_[input_offset + i] =
          GetBinIdx(gch_options_, item_hash, items_hash_u64.first);
    }
  });

  for (size_t i = 0; i < items.size(); ++i) {
    Insert(items[i], input_offset + i);
  }
}

SimpleHashTable::SimpleHashTable(CuckooIndex::Options options, uint128_t seed)
    : gch_options_(options), seed_(seed) {
  size_t table_size = gch_options_.NumBins();

  bins_.resize(table_size);
}

void SimpleHashTable::Insert(yacl::ByteContainerView item_data,
                             const std::vector<uint64_t> &hash_bin_idx) {
  CuckooIndex::Bin candidate;
  candidate.set_encoded(inserted_items_);

  std::set<size_t> idx_set(hash_bin_idx.begin(), hash_bin_idx.end());

  size_t bin_idx;

  if (idx_set.size() < hash_bin_idx.size()) {
    SPDLOG_WARN("hash conflict: bin_idx:{}, data:{}", hash_bin_idx[0],
                absl::BytesToHexString(absl::string_view(
                    reinterpret_cast<const char *>(item_data.data()),
                    item_data.size())));

    conflict_idx_.push_back(candidate.InputIdx());

    bin_idx = hash_bin_idx[0];
    uint64_t next_candid = CuckooIndex::Bin::Encode(candidate.InputIdx(), 0);
    candidate.set_encoded(next_candid);
    bins_[bin_idx].push_back(candidate);

  } else {
    for (size_t i = 0; i < gch_options_.num_hash; i++) {
      bin_idx = hash_bin_idx[i];

      uint64_t next_candid = CuckooIndex::Bin::Encode(candidate.InputIdx(), i);
      candidate.set_encoded(next_candid);
      bins_[bin_idx].push_back(candidate);
    }
  }

  inserted_items_++;
}

void SimpleHashTable::Insert(yacl::ByteContainerView item) {
  uint128_t item_hash = yacl::crypto::Blake3_128(item);

  std::vector<uint64_t> hash_bin_idx;
  std::pair<uint64_t, uint64_t> item_hash_u64 =
      yacl::DecomposeUInt128(item_hash);

  items_hash_low64_.push_back(item_hash_u64.second);

  hash_bin_idx = GetBinIdx(gch_options_, item_hash, item_hash_u64.first);

  Insert(item, hash_bin_idx);
}

void SimpleHashTable::Insert(absl::Span<const std::string> items) {
  size_t input_offset = items_hash_low64_.size();
  items_hash_low64_.resize(input_offset + items.size());

  std::vector<std::vector<uint64_t>> hash_bin_idx(items.size());

  yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    for (int i = begin; i < end; ++i) {
      uint128_t item_hash = yacl::crypto::Blake3_128(items[i]);

      std::pair<uint64_t, uint64_t> item_hash_u64 =
          yacl::DecomposeUInt128(item_hash);
      hash_bin_idx[i] = GetBinIdx(gch_options_, item_hash, item_hash_u64.first);

      items_hash_low64_[input_offset + i] = item_hash_u64.second;
    }
  });

  for (size_t i = 0; i < items.size(); ++i) {
    Insert(items[i], hash_bin_idx[i]);
  }
}

}  // namespace spu::psi
