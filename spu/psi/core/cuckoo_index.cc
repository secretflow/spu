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

#include "spu/psi/core/cuckoo_index.h"

#include <cmath>
#include <set>

namespace spu::psi {

CuckooIndex::CuckooIndex(const Options& options) : options_(options) {
  bins_.resize(options_.NumBins());
  stash_.resize(options_.num_stash);
  hashes_.reserve(options_.NumBins());

  YASL_ENFORCE((options_.num_hash - 1) * HashRoom::kBlockSize +
                   sizeof(uint64_t) <=
               sizeof(HashType));
}

void CuckooIndex::Insert(absl::Span<const HashType> codes) {
  const size_t input_offset = hashes_.size();
  const size_t size = codes.size();
  const size_t num_bins = options_.NumBins();

  // Add to hash rooms.
  for (const HashType& code : codes) {
    hashes_.push_back(HashRoom(code));
  }
  YASL_ENFORCE(hashes_.size() <= options_.num_input);

  std::vector<Bin> candidates(size);
  for (size_t i = 0; i < candidates.size(); ++i) {
    // Init candidates. Start from first hash function.
    candidates[i].set_encoded(input_offset + i);
  }

  size_t try_count = 0;
  std::vector<Bin> evicted;

  while (!candidates.empty() && try_count++ < options_.max_try_count) {
    size_t write_idx = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Bin candid = candidates[i];
      size_t bin_idx =
          hashes_[candid.InputIdx()].GetHash(candid.HashIdx()) % num_bins;
      Bin evicted_bin = Bin(bins_[bin_idx].Swap(candid.encoded()));
      if (!evicted_bin.IsEmpty()) {
        // Try next hash for evicted items.
        uint64_t next_candid =
            Bin::Encode(evicted_bin.InputIdx(),
                        (evicted_bin.HashIdx() + 1) % options_.num_hash);
        candidates[write_idx++].set_encoded(next_candid);
      }
    }
    candidates.resize(write_idx);
  }

  for (size_t i = 0; i < candidates.size(); ++i) {
    PutToStash(candidates[i].InputIdx());
  }
}

void CuckooIndex::PutToStash(uint64_t input_idx) {
  // `stash` is small enough to do a linear search.
  for (size_t i = 0; i < stash_.size(); ++i) {
    if (stash_[i].IsEmpty()) {
      stash_[i].set_encoded(input_idx);
      return;
    }
  }
  YASL_THROW("Cannot find empty bin in stash for input_idx={}", input_idx);
}

void CuckooIndex::SanityCheck() const {
  std::set<uint64_t> set;
  for (const auto& bin : bins_) {
    if (!bin.IsEmpty()) {
      YASL_ENFORCE(set.insert(bin.InputIdx()).second,
                   "Input={} already exists.", bin.InputIdx());
    }
  }
  for (const auto& bin : stash_) {
    if (!bin.IsEmpty()) {
      YASL_ENFORCE(set.insert(bin.InputIdx()).second,
                   "Input={} already exists.", bin.InputIdx());
    }
  }

  // All inputs should be found.
  YASL_ENFORCE(set.size() == options_.num_input);
  // Every input must exists.
  size_t idx = 0;
  for (uint64_t i : set) {
    YASL_ENFORCE(idx++ == i, "Cannot find input={}", i);
  }
}

// computing method from cryptoTool
// https://github.com/ladnir/cryptoTools/blob/master/cryptoTools/Common/CuckooIndex.cpp#L133
CuckooIndex::Options CuckooIndex::SelectParams(uint64_t n, uint64_t stash_size,
                                               uint64_t hash_num,
                                               uint64_t stat_sec_param) {
  auto h = hash_num ? hash_num : 3;

  if (stash_size == 0 && h == 3) {
    double a = 240;
    double b = -std::log2(n) - 256;

    auto e = (stat_sec_param - b) / a;

    // we have the statSecParam = a e + b, where e = |cuckoo|/|set| is the
    // expenation factor therefore we have that
    //
    //   e = (statSecParam - b) / a
    //

    return CuckooIndex::Options{n, 0, h, e};
  }

  YASL_THROW("not support for stash_size={} and hash_num={}", stash_size,
             hash_num);
}

uint8_t CuckooIndex::MinCollidingHashIdx(uint64_t bin_index) const {
  auto bin = bins_[bin_index];
  size_t num_bins = options_.NumBins();
  uint64_t input_idx = bin.InputIdx();
  uint64_t target;
  for (uint64_t i = 0; i < options_.num_hash; i++) {
    target = hashes_[input_idx].GetHash(i) % num_bins;
    if (target == bin_index) {
      return uint8_t(i);
    }
  }
  return -1;
}

}  // namespace spu::psi