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

#include "libspu/psi/utils/cipher_store.h"

#include <algorithm>
#include <future>
#include <unordered_set>

#include "absl/strings/escaping.h"
#include "spdlog/spdlog.h"

#include "libspu/psi/utils/batch_provider.h"

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

CachedCsvCipherStore::CachedCsvCipherStore(const std::string& self_csv,
                                           const std::string& peer_csv,
                                           bool self_read_only,
                                           bool peer_read_only)
    : self_csv_path_(self_csv),
      peer_csv_path_(peer_csv),
      self_read_only_(self_read_only),
      peer_read_only_(peer_read_only) {
  if (!self_read_only_) {
    self_out_ = io::BuildOutputStream(io::FileIoOptions(self_csv_path_));
    self_out_->Write("id\n");
  }
  if (!peer_read_only_) {
    peer_out_ = io::BuildOutputStream(io::FileIoOptions(peer_csv_path_));
    peer_out_->Write("id\n");
  }
}

CachedCsvCipherStore::~CachedCsvCipherStore() {
  if (!self_read_only_) {
    self_out_->Close();
  }

  if (!peer_read_only_) {
    peer_out_->Close();
  }
}

void CachedCsvCipherStore::SaveSelf(std::string ciphertext) {
  std::string hex_str = absl::BytesToHexString(ciphertext);
  self_out_->Write(fmt::format("{}\n", hex_str));
  self_data_.push_back(hex_str);

  self_items_count_ += 1;
  if (self_items_count_ % 10000000 == 0) {
    SPDLOG_INFO("self_items_count_={}", self_items_count_);
  }
}

void CachedCsvCipherStore::SavePeer(std::string ciphertext) {
  peer_out_->Write(fmt::format("{}\n", absl::BytesToHexString(ciphertext)));
  peer_items_count_ += 1;
  if (peer_items_count_ % 10000000 == 0) {
    SPDLOG_INFO("peer_items_count={}", peer_items_count_);
  }
}

void CachedCsvCipherStore::SaveSelf(
    const std::vector<std::string>& ciphertext) {
  for (size_t i = 0; i < ciphertext.size(); i++) {
    std::string hex_str = absl::BytesToHexString(ciphertext[i]);
    self_out_->Write(fmt::format("{}\n", hex_str));
    self_data_.push_back(hex_str);

    self_items_count_ += 1;
    if (self_items_count_ % 10000000 == 0) {
      SPDLOG_INFO("self_items_count_={}", self_items_count_);
    }
  }
}

void CachedCsvCipherStore::SavePeer(
    const std::vector<std::string>& ciphertext) {
  for (size_t i = 0; i < ciphertext.size(); i++) {
    peer_out_->Write(
        fmt::format("{}\n", absl::BytesToHexString(ciphertext[i])));

    peer_items_count_ += 1;
    if (peer_items_count_ % 10000000 == 0) {
      SPDLOG_INFO("peer_items_count={}", peer_items_count_);
    }
  }
}

std::vector<uint64_t> CachedCsvCipherStore::FinalizeAndComputeIndices(
    size_t bucket_size) {
  if (!self_read_only_) {
    self_out_->Flush();
  }
  FlushPeer();

  SPDLOG_INFO("Begin FinalizeAndComputeIndices");

  std::vector<uint64_t> indices;

  std::vector<std::string> ids = {"id"};
  CsvBatchProvider peer_provider(peer_csv_path_, ids);
  size_t batch_count = 0;

  while (true) {
    SPDLOG_INFO("begin read compare batch {}", batch_count);
    std::vector<std::string> batch_peer_data =
        peer_provider.ReadNextBatch(bucket_size);
    SPDLOG_INFO("end read compare batch {}", batch_count);

    if (batch_peer_data.empty()) {
      break;
    }

    size_t compare_size =
        (self_data_.size() + compare_thread_num_ - 1) / compare_thread_num_;

    std::vector<std::vector<uint64_t>> result(compare_thread_num_);

    auto compare_proc = [&](int idx) -> void {
      uint64_t begin = idx * compare_size;
      uint64_t end = std::min<size_t>(self_data_.size(), begin + compare_size);

      for (uint64_t i = begin; i < end; i++) {
        if (std::binary_search(batch_peer_data.begin(), batch_peer_data.end(),
                               self_data_[i])) {
          result[idx].push_back(i);
        }
      }
    };

    std::vector<std::future<void>> f_compare(compare_thread_num_);
    for (size_t i = 0; i < compare_thread_num_; i++) {
      f_compare[i] = std::async(compare_proc, i);
    }

    for (size_t i = 0; i < compare_thread_num_; i++) {
      f_compare[i].get();
    }

    batch_count++;
    for (size_t i = 0; i < result.size(); ++i) {
      indices.insert(indices.end(), result[i].begin(), result[i].end());
    }
    SPDLOG_INFO("FinalizeAndComputeIndices, batch_count:{}", batch_count);
  }

  std::sort(indices.begin(), indices.end());
  SPDLOG_INFO("End FinalizeAndComputeIndices, batch_count:{}", batch_count);
  return indices;
}

}  // namespace spu::psi
