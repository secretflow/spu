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

#include <memory>
#include <string>
#include <vector>

#include "spu/psi/utils/hash_bucket_cache.h"

namespace spu::psi {

/// ICipherStore stores dual encrypted results.
class ICipherStore {
 public:
  virtual ~ICipherStore() = default;

  // SaveSelf/SavePeer saves the dual encrypted ciphertext.
  //
  // Threading:
  // Each function is guaranteed to be called in one thread during the
  // `RunEcdhPsi`. However, the caller threads for these two functions are
  // different.
  //
  // Order:
  // The save order is same as the input order provided by `IBatchProvider`.
  //
  // Contraint:
  // The two functions wont be called by `RunEcdhPsi` if my rank does not
  // match the `target_rank`.
  virtual void SaveSelf(std::string ciphertext) = 0;
  virtual void SavePeer(std::string ciphertext) = 0;
};

class MemoryCipherStore : public ICipherStore {
 public:
  void SaveSelf(std::string ciphertext) override {
    self_results_.push_back(std::move(ciphertext));
  }

  void SavePeer(std::string ciphertext) override {
    peer_results_.push_back(std::move(ciphertext));
  }

  const std::vector<std::string>& self_results() const { return self_results_; }
  const std::vector<std::string>& peer_results() const { return peer_results_; }

  std::vector<std::string>& self_results() { return self_results_; }
  std::vector<std::string>& peer_results() { return peer_results_; }

 private:
  std::vector<std::string> self_results_;
  std::vector<std::string> peer_results_;
};

class DiskCipherStore : public ICipherStore {
 public:
  explicit DiskCipherStore(const std::string& cache_dir, size_t num_bins);

  void SaveSelf(std::string ciphertext) override;

  void SavePeer(std::string ciphertext) override;

  std::vector<uint64_t> FinalizeAndComputeIndices();

 private:
  void FindIntersectionIndices(size_t bucket_idx,
                               std::vector<uint64_t>* indices);

 private:
  const size_t num_bins_;

  std::unique_ptr<HashBucketCache> self_cache_;
  std::unique_ptr<HashBucketCache> peer_cache_;
};

}  // namespace spu::psi
