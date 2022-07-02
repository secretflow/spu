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

#include <vector>

#include "absl/types/span.h"
#include "yasl/base/exception.h"
#include "yasl/base/int128.h"

namespace spu::psi {

// CuckooIndex does not want to be a container like `unordered_map` which
// provides CRUD interfaces. Instead, CuckooIndex aims to decide the location
// for each input. And it is widely used in PSI or PIR for a perfect `1 to N`
// relationships.
//
// Reference:
// https://github.com/ladnir/cryptoTools/blob/924e328071b8c0616df01a233705702a7f6df2c3/cryptoTools/Common/CuckooIndex.h
//
// This class give a slightly more cpp-style interface and it's more readable.
class CuckooIndex {
 public:
  // Leave a space for us to adjust hash algorithm later, say `blake3` will
  // output 2356 bits.
  using HashType = uint128_t;

  // Compact bin representation.
  //
  // Sample: this bin stores input `input_idx=2` via the second hash function
  // (hash_idx=1).
  //
  // |----------uint64_t---------------------|
  // |  8-bit  |        56-bit               |
  // | hashidx |     input idx               |
  //   00000001_0000000000000...0000000000010
  // |---------------------------------------|
  class Bin {
   public:
    inline static constexpr uint64_t kEmpty = uint64_t(-1);

    explicit Bin(uint64_t encoded = kEmpty) : encoded_(encoded) {}

    uint8_t HashIdx() const { return uint8_t(encoded_ >> 56); }
    uint64_t InputIdx() const { return encoded_ & (uint64_t(-1) >> 8); }

    uint64_t encoded() const { return encoded_; }
    void set_encoded(uint64_t encoded) { encoded_ = encoded; }

    uint64_t Swap(uint64_t encoded) {
      uint64_t prev = encoded_;
      encoded_ = encoded;
      return prev;
    }

    bool IsEmpty() const { return encoded_ == kEmpty; }

    static uint64_t Encode(uint64_t input_idx, uint8_t hash_idx) {
      return input_idx | uint64_t(hash_idx) << 56;
    }

   private:
    uint64_t encoded_;
  };

  // Compact hash codes store.
  class HashRoom {
   public:
    explicit HashRoom(HashType code) : code_(code) {}

    inline static constexpr int kBlockSize = 2;

    uint64_t GetHash(size_t idx) const {
      // Make it portable on non-x86 archs.
      // References:
      //  https://blog.quarkslab.com/unaligned-accesses-in-cc-what-why-and-solutions-to-do-it-properly.html
      //  https://www.kernel.org/doc/Documentation/unaligned-memory-access.txt
      uint64_t aligned_u64;
      memcpy(&aligned_u64, (const uint8_t*)(&code_) + idx * kBlockSize,
             sizeof(aligned_u64));
      return aligned_u64;
    }

   private:
    const HashType code_;
  };

  struct Options {
    // Number of expected inputs.
    uint64_t num_input;
    // Stash capcity.
    uint64_t num_stash;
    // Number of hash functions used.
    uint64_t num_hash;
    // Scale factor for computing `NumBins`.
    double scale_factor;
    // Decide the max number of evictions allowed for each `Insert`. These
    // evicted items will be put into `stash` finally. If the stash is full, an
    // exception will ocurred.
    uint64_t max_try_count = 128;

    uint64_t NumBins() const {
      uint64_t num_bins = num_input * scale_factor;
      //
      // for stashless cuckooHash
      // when num_input < 256, num_bins add  8 for safe
      // reduce the probability of using stash
      if ((num_stash == 0) && (num_input < 256)) {
        num_bins += 8;
      }
      return num_bins;
    }
  };

 public:
  explicit CuckooIndex(const Options& options);

  const std::vector<Bin>& bins() const { return bins_; }

  const std::vector<Bin>& stash() const { return stash_; }

  const std::vector<HashRoom>& hashes() const { return hashes_; }

  // This interface assumes `inputs` are already cryptographic random.
  void Insert(absl::Span<const HashType> codes);

  // For debug only.
  void SanityCheck() const;

  static Options SelectParams(uint64_t n, uint64_t stash_size,
                              uint64_t hash_num, uint64_t stat_sec_param = 40);

  uint8_t MinCollidingHashIdx(uint64_t bin_index) const;

 private:
  void PutToStash(uint64_t input_idx);

  const Options options_;
  std::vector<Bin> bins_;
  std::vector<Bin> stash_;
  std::vector<HashRoom> hashes_;
};

}  // namespace spu::psi
