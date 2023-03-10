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

#include <omp.h>

#include <algorithm>
#include <future>
#include <unordered_set>
#include <utility>

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

CachedCsvCipherStore::CachedCsvCipherStore(std::string self_csv,
                                           std::string peer_csv,
                                           bool self_read_only,
                                           bool peer_read_only)
    : self_csv_path_(std::move(self_csv)),
      peer_csv_path_(std::move(peer_csv)),
      self_read_only_(self_read_only),
      peer_read_only_(peer_read_only) {
  if (!self_read_only_) {
    self_out_ = io::BuildOutputStream(io::FileIoOptions(self_csv_path_));
    self_out_->Write(fmt::format("{}\n", cipher_id_));
  }
  if (!peer_read_only_) {
    peer_out_ = io::BuildOutputStream(io::FileIoOptions(peer_csv_path_));
    peer_out_->Write(fmt::format("{}\n", cipher_id_));
  }

  compare_thread_num_ = omp_get_num_procs();
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
  self_out_->Write(fmt::format("{}\n", ciphertext));
  self_data_.insert({ciphertext, self_items_count_});

  self_items_count_ += 1;
  if (self_items_count_ % 10000000 == 0) {
    SPDLOG_INFO("self_items_count_={}", self_items_count_);
  }
}

void CachedCsvCipherStore::SavePeer(std::string ciphertext) {
  peer_out_->Write(fmt::format("{}\n", ciphertext));
  peer_items_count_ += 1;
  if (peer_items_count_ % 10000000 == 0) {
    SPDLOG_INFO("peer_items_count={}", peer_items_count_);
  }
}

void CachedCsvCipherStore::SaveSelf(
    const std::vector<std::string>& ciphertext) {
  for (const auto& text : ciphertext) {
    self_out_->Write(fmt::format("{}\n", text));
    self_data_.insert({text, self_items_count_});

    self_items_count_ += 1;
    if (self_items_count_ % 10000000 == 0) {
      SPDLOG_INFO("self_items_count_={}", self_items_count_);
    }
  }
}

void CachedCsvCipherStore::SavePeer(
    const std::vector<std::string>& ciphertext) {
  for (const auto& text : ciphertext) {
    peer_out_->Write(fmt::format("{}\n", text));

    peer_items_count_ += 1;
    if (peer_items_count_ % 10000000 == 0) {
      SPDLOG_INFO("peer_items_count={}", peer_items_count_);
    }
  }
}

std::pair<std::vector<uint64_t>, std::vector<std::string>>
CachedCsvCipherStore::FinalizeAndComputeIndices(size_t bucket_size) {
  if (!self_read_only_) {
    self_out_->Flush();
  }
  FlushPeer();

  SPDLOG_INFO("Begin FinalizeAndComputeIndices");

  std::vector<uint64_t> indices;
  std::vector<std::string> masked_items;

  std::vector<std::string> ids = {cipher_id_};
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

    size_t compare_size = (batch_peer_data.size() + compare_thread_num_ - 1) /
                          compare_thread_num_;

    std::vector<std::vector<uint64_t>> batch_indices(compare_thread_num_);
    std::vector<std::vector<std::string>> batch_masked_items(
        compare_thread_num_);

    auto compare_proc = [&](int idx) -> void {
      uint64_t begin = idx * compare_size;
      uint64_t end =
          std::min<uint64_t>(batch_peer_data.size(), begin + compare_size);

      for (size_t i = begin; i < end; i++) {
        auto search_ret = self_data_.find(batch_peer_data[i]);
        if (search_ret != self_data_.end()) {
          batch_indices[idx].push_back(search_ret->second);
          batch_masked_items[idx].push_back(batch_peer_data[i]);
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

    for (const auto& r : batch_indices) {
      indices.insert(indices.end(), r.begin(), r.end());
    }
    for (const auto& r : batch_masked_items) {
      masked_items.insert(masked_items.end(), r.begin(), r.end());
    }
    SPDLOG_INFO("FinalizeAndComputeIndices, batch_count:{}", batch_count);
  }

  SPDLOG_INFO("End FinalizeAndComputeIndices, batch_count:{}", batch_count);
  return std::make_pair(indices, masked_items);
}

std::vector<uint64_t> GetIndicesByItems(
    const std::string& input_path,
    const std::vector<std::string>& selected_fields,
    const std::vector<std::string>& items, size_t batch_size) {
  std::vector<uint64_t> indices;

  std::unordered_set<std::string> items_set;

  items_set.insert(items.begin(), items.end());

  std::shared_ptr<IBatchProvider> batch_provider =
      std::make_shared<CsvBatchProvider>(input_path, selected_fields);

  size_t compare_thread_num = omp_get_num_procs();

  size_t item_index = 0;
  size_t batch_count = 0;
  while (true) {
    auto batch_items = batch_provider->ReadNextBatch(batch_size);
    if (batch_items.empty()) {
      break;
    }

    size_t compare_size =
        (batch_items.size() + compare_thread_num - 1) / compare_thread_num;

    std::vector<std::vector<uint64_t>> result(compare_thread_num);

    auto compare_proc = [&](int idx) -> void {
      uint64_t begin = idx * compare_size;
      uint64_t end = std::min<size_t>(batch_items.size(), begin + compare_size);

      for (uint64_t i = begin; i < end; i++) {
        auto search_ret = items_set.find(batch_items[i]);
        if (search_ret != items_set.end()) {
          result[idx].push_back(item_index + i);
        }
      }
    };

    std::vector<std::future<void>> f_compare(compare_thread_num);
    for (size_t i = 0; i < compare_thread_num; i++) {
      f_compare[i] = std::async(compare_proc, i);
    }

    for (size_t i = 0; i < compare_thread_num; i++) {
      f_compare[i].get();
    }

    for (const auto& r : result) {
      indices.insert(indices.end(), r.begin(), r.end());
    }

    batch_count++;

    item_index += batch_items.size();
    SPDLOG_INFO("GetIndices batch count:{}, item_index:{}", batch_count,
                item_index);
  }
  SPDLOG_INFO("Finish GetIndices, indices size:{}", indices.size());

  return indices;
}

}  // namespace spu::psi
