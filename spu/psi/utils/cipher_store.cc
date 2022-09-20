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

#include "spu/psi/utils/cipher_store.h"

#include <unordered_set>

#include "spdlog/spdlog.h"

namespace spu::psi {

DiskCipherStore::DiskCipherStore(const std::string& cache_dir, size_t num_bins)
    : num_bins_(std::max(1UL, num_bins)) {
  SPDLOG_INFO("Disk cache choose num_bins={}", num_bins_);

  self_cache_ = std::make_unique<HashBucketCache>(cache_dir, num_bins_);
  peer_cache_ = std::make_unique<HashBucketCache>(cache_dir, num_bins_);
}

void DiskCipherStore::SaveSelf(std::string ciphertext) {
  self_cache_->WriteItem(ciphertext);
}

void DiskCipherStore::SavePeer(std::string ciphertext) {
  peer_cache_->WriteItem(ciphertext);
}

std::vector<uint64_t> DiskCipherStore::FinalizeAndComputeIndices() {
  self_cache_->Flush();
  peer_cache_->Flush();

  // Compute indices
  std::vector<uint64_t> indices;
  for (size_t bin_idx = 0; bin_idx < num_bins_; ++bin_idx) {
    FindIntersectionIndices(bin_idx, &indices);
  }
  // Sort to make `FilterByLine` happy.
  std::sort(indices.begin(), indices.end());
  return indices;
}

void DiskCipherStore::FindIntersectionIndices(size_t bucket_idx,
                                              std::vector<uint64_t>* indices) {
  std::vector<HashBucketCache::BucketItem> self_results =
      self_cache_->LoadBucketItems(bucket_idx);
  std::vector<HashBucketCache::BucketItem> peer_results =
      peer_cache_->LoadBucketItems(bucket_idx);
  std::unordered_set<std::string> peer_set;
  peer_set.reserve(peer_results.size());
  std::for_each(peer_results.begin(), peer_results.end(),
                [&](HashBucketCache::BucketItem& item) {
                  peer_set.insert(std::move(item.base64_data));
                });
  for (const auto& item : self_results) {
    if (peer_set.find(item.base64_data) != peer_set.end()) {
      indices->push_back(item.index);
    }
  }
}

}  // namespace spu::psi