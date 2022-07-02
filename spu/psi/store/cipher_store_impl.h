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

#include "spu/psi/io/io.h"
#include "spu/psi/store/cipher_store.h"
#include "spu/psi/store/scope_disk_cache.h"

namespace spu::psi {

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

// TODO: refactor
class DiskCipherStore : public ICipherStore {
 public:
  struct ItemWithIndex {
    std::string item;
    unsigned index;

    bool operator<(const ItemWithIndex& other) const {
      return item < other.item;
    }
    bool operator==(const ItemWithIndex& other) const {
      return item == other.item;
    }
  };

 public:
  explicit DiskCipherStore(const std::string& cache_dir, size_t num_bins);

  void SaveSelf(std::string ciphertext) override;

  void SavePeer(std::string ciphertext) override;

  void Finalize();

  std::vector<unsigned> FinalizeAndComputeIndices();

  std::vector<ItemWithIndex> LoadSelfBinFile(size_t bin_idx);

  std::vector<ItemWithIndex> LoadPeerBinFile(size_t bin_idx);

 private:
  std::vector<ItemWithIndex> LoadBinFile(const std::string& path);

  void FindIntersectionIndices(const std::string& self_path,
                               const std::string& peer_path,
                               std::vector<unsigned>* indices);

  void CreateDiskCache(
      const std::string& cache_dir, size_t num_bins,
      std::unique_ptr<ScopeDiskCache>* out_cache,
      std::vector<std::unique_ptr<io::OutputStream>>* out_streams);

 private:
  const size_t num_bins_;
  std::unique_ptr<ScopeDiskCache> self_disk_cache_;
  std::unique_ptr<ScopeDiskCache> peer_disk_cache_;
  std::vector<std::unique_ptr<io::OutputStream>> self_stores_;
  std::vector<std::unique_ptr<io::OutputStream>> peer_stores_;

  size_t self_store_index_ = 0;
  size_t peer_store_index_ = 0;
};

}  // namespace spu::psi