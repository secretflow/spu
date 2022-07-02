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

#include "spu/psi/store/cipher_store_impl.h"

#include <filesystem>
#include <unordered_set>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "spdlog/spdlog.h"

namespace spu::psi {

DiskCipherStore::DiskCipherStore(const std::string& cache_dir, size_t num_bins)
    : num_bins_(std::max(1UL, num_bins)) {
  SPDLOG_INFO("Disk cache choose num_bins={}", num_bins_);

  // Create self disk cache and file sinks.
  CreateDiskCache(cache_dir, num_bins_, &self_disk_cache_, &self_stores_);
  CreateDiskCache(cache_dir, num_bins_, &peer_disk_cache_, &peer_stores_);
}

void DiskCipherStore::SaveSelf(std::string ciphertext) {
  std::string base64_item = absl::Base64Escape(ciphertext);
  auto& out =
      self_stores_[std::hash<std::string>()(base64_item) % self_stores_.size()];
  out->Write(base64_item);
  out->Write(",");
  out->Write(std::to_string(self_store_index_));
  out->Write("\n");
  self_store_index_++;
}

void DiskCipherStore::SavePeer(std::string ciphertext) {
  std::string base64_item = absl::Base64Escape(ciphertext);
  auto& out =
      peer_stores_[std::hash<std::string>()(base64_item) % peer_stores_.size()];
  out->Write(base64_item);
  out->Write(",");
  out->Write(std::to_string(peer_store_index_));
  out->Write("\n");
  peer_store_index_++;
}

void DiskCipherStore::Finalize() {
  // Flush files.
  for (auto& out : self_stores_) {
    out->Close();
  }
  for (auto& out : peer_stores_) {
    out->Close();
  }
}

std::vector<unsigned> DiskCipherStore::FinalizeAndComputeIndices() {
  // Flush files.
  for (auto& out : self_stores_) {
    out->Close();
  }
  for (auto& out : peer_stores_) {
    out->Close();
  }

  // Compute indices
  std::vector<unsigned> indices;
  for (size_t bin_idx = 0; bin_idx < num_bins_; ++bin_idx) {
    FindIntersectionIndices(self_disk_cache_->GetBinPath(bin_idx),
                            peer_disk_cache_->GetBinPath(bin_idx), &indices);
  }
  // Sort to make `FilterByLine` happy.
  std::sort(indices.begin(), indices.end());
  return indices;
}

std::vector<DiskCipherStore::ItemWithIndex> DiskCipherStore::LoadSelfBinFile(
    size_t bin_idx) {
  return LoadBinFile(self_disk_cache_->GetBinPath(bin_idx));
}

std::vector<DiskCipherStore::ItemWithIndex> DiskCipherStore::LoadPeerBinFile(
    size_t bin_idx) {
  return LoadBinFile(peer_disk_cache_->GetBinPath(bin_idx));
}

std::vector<DiskCipherStore::ItemWithIndex> DiskCipherStore::LoadBinFile(
    const std::string& path) {
  auto in = io::BuildInputStream(io::FileIoOptions(path));

  std::string line;
  DiskCipherStore::ItemWithIndex item_with_index;
  std::vector<DiskCipherStore::ItemWithIndex> ret;
  while (in->GetLine(&line)) {
    std::vector<absl::string_view> tokens = absl::StrSplit(line, ",");
    YASL_ENFORCE(tokens.size() == 2, "Should have two tokens, actual: {}",
                 tokens.size());
    item_with_index.item = std::string(tokens[0].data(), tokens[0].size());
    YASL_ENFORCE(absl::SimpleAtoi(tokens[1], &item_with_index.index),
                 "Cannot convert to idx: {}",
                 std::string(tokens[1].data(), tokens[1].size()));
    ret.push_back(std::move(item_with_index));
  }
  return ret;
}

void DiskCipherStore::FindIntersectionIndices(const std::string& self_path,
                                              const std::string& peer_path,
                                              std::vector<unsigned>* indices) {
  std::vector<DiskCipherStore::ItemWithIndex> self_results =
      LoadBinFile(self_path);
  std::vector<DiskCipherStore::ItemWithIndex> peer_results =
      LoadBinFile(peer_path);
  std::unordered_set<std::string> peer_set;
  peer_set.reserve(peer_results.size());
  std::for_each(peer_results.begin(), peer_results.end(),
                [&](DiskCipherStore::ItemWithIndex& pair) {
                  peer_set.insert(std::move(pair.item));
                });
  for (const auto& item_with_index : self_results) {
    if (peer_set.find(item_with_index.item) != peer_set.end()) {
      indices->push_back(item_with_index.index);
    }
  }
}

void DiskCipherStore::CreateDiskCache(
    const std::string& cache_dir, size_t num_bins,
    std::unique_ptr<ScopeDiskCache>* out_cache,
    std::vector<std::unique_ptr<io::OutputStream>>* out_streams) {
  *out_cache = ScopeDiskCache::Create(std::filesystem::path(cache_dir));
  YASL_ENFORCE(*out_cache, "Cannot create disk cache from dir={}", cache_dir);
  (*out_cache)->CreateHashBinStreams(num_bins, out_streams);
}

}  // namespace spu::psi