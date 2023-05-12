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

#include "libspu/psi/utils/batch_provider.h"

#include <algorithm>
#include <future>
#include <random>

#include "absl/strings/escaping.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/utils.h"

namespace spu::psi {

MemoryBatchProvider::MemoryBatchProvider(const std::vector<std::string>& items,
                                         bool shuffle)
    : items_(items), shuffle_(shuffle), labels_(items) {
  shuffled_indices_.resize(items.size());
  std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);

  if (shuffle_) {
    std::mt19937 rng(yacl::crypto::SecureRandU64());
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), rng);
  }
  is_labeled_ = true;
}

std::vector<std::string> MemoryBatchProvider::ReadNextBatch(size_t batch_size) {
  std::vector<std::string> batch;
  SPU_ENFORCE(cursor_index_ <= items_.size());
  size_t n_items = std::min(batch_size, items_.size() - cursor_index_);
  batch.insert(batch.end(), items_.begin() + cursor_index_,
               items_.begin() + cursor_index_ + n_items);
  cursor_index_ += n_items;
  return batch;
}

std::pair<std::vector<std::string>, std::vector<std::string>>
MemoryBatchProvider::ReadNextBatchWithLabel(size_t batch_size) {
  std::vector<std::string> batch_items;
  std::vector<std::string> batch_labels;

  if (!is_labeled_) {
    return std::make_pair(batch_items, batch_labels);
  }

  SPU_ENFORCE(cursor_index_ <= items_.size());
  size_t n_items = std::min(batch_size, items_.size() - cursor_index_);

  batch_items.insert(batch_items.end(), items_.begin() + cursor_index_,
                     items_.begin() + cursor_index_ + n_items);

  batch_labels.insert(batch_labels.end(), labels_.begin() + cursor_index_,
                      labels_.begin() + cursor_index_ + n_items);

  cursor_index_ += n_items;
  return std::make_pair(batch_items, batch_labels);
}

std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>>
MemoryBatchProvider::ReadNextBatchWithIndex(size_t batch_size) {
  std::vector<std::string> batch_data;
  std::vector<size_t> batch_indices;
  std::vector<size_t> shuffle_indices;

  SPU_ENFORCE(cursor_index_ <= items_.size());
  size_t n_items = std::min(batch_size, items_.size() - cursor_index_);
  for (size_t i = 0; i < n_items; ++i) {
    size_t shuffled_index = shuffled_indices_[cursor_index_ + i];
    batch_data.push_back(items_[shuffled_index]);
    batch_indices.push_back(cursor_index_ + i);
    shuffle_indices.push_back(shuffled_index);
  }

  cursor_index_ += n_items;

  return std::make_tuple(batch_data, batch_indices, shuffle_indices);
}

const std::vector<std::string>& MemoryBatchProvider::items() const {
  return items_;
}

const std::vector<std::string>& MemoryBatchProvider::labels() const {
  if (is_labeled_) {
    return labels_;
  } else {
    SPU_THROW("Not in Labeled model");
  }
}

const std::vector<size_t>& MemoryBatchProvider::shuffled_indices() const {
  return shuffled_indices_;
}

CsvBatchProvider::CsvBatchProvider(
    const std::string& path, const std::vector<std::string>& target_fields)
    : path_(path),
      analyzer_(path, target_fields),
      label_analyzer_(path, target_fields) {
  in_ = io::BuildInputStream(io::FileIoOptions(path_));
  // skip header
  std::string line;
  in_->GetLine(&line);
}

CsvBatchProvider::CsvBatchProvider(const std::string& path,
                                   const std::vector<std::string>& item_fields,
                                   const std::vector<std::string>& label_fields)
    : path_(path),
      analyzer_(path, item_fields),
      label_analyzer_(path, label_fields) {
  in_ = io::BuildInputStream(io::FileIoOptions(path_));
  // skip header
  std::string line;
  in_->GetLine(&line);

  is_labeled_ = true;
}

std::vector<std::string> CsvBatchProvider::ReadNextBatch(size_t batch_size) {
  std::vector<std::string> ret;
  std::string line;
  while (in_->GetLine(&line)) {
    std::vector<absl::string_view> tokens = absl::StrSplit(line, ',');
    std::vector<absl::string_view> targets;
    for (size_t fidx : analyzer_.target_indices()) {
      SPU_ENFORCE(fidx < tokens.size(),
                  "Illegal line due to no field at index={}, line={}", fidx,
                  line);
      targets.push_back(absl::StripAsciiWhitespace(tokens[fidx]));
    }
    ret.push_back(KeysJoin(targets));
    if (ret.size() == batch_size) {
      break;
    }
  }
  return ret;
}

std::pair<std::vector<std::string>, std::vector<std::string>>
CsvBatchProvider::ReadNextBatchWithLabel(size_t batch_size) {
  std::pair<std::vector<std::string>, std::vector<std::string>> ret;
  std::string line;

  if (!is_labeled_) {
    return ret;
  }

  while (in_->GetLine(&line)) {
    std::vector<absl::string_view> tokens = absl::StrSplit(line, ',');
    std::vector<absl::string_view> items;
    std::vector<absl::string_view> labels;
    for (size_t fidx : analyzer_.target_indices()) {
      SPU_ENFORCE(fidx < tokens.size(),
                  "Illegal line due to no field at index={}, line={}", fidx,
                  line);
      items.push_back(absl::StripAsciiWhitespace(tokens[fidx]));
    }
    for (size_t fidx : label_analyzer_.target_indices()) {
      SPU_ENFORCE(fidx < tokens.size(),
                  "Illegal line due to no field at index={}, line={}", fidx,
                  line);
      labels.push_back(absl::StripAsciiWhitespace(tokens[fidx]));
    }

    ret.first.push_back(KeysJoin(items));
    ret.second.push_back(KeysJoin(labels));

    if (ret.first.size() == batch_size) {
      break;
    }
  }

  return ret;
}

CachedCsvBatchProvider::CachedCsvBatchProvider(
    const std::string& path, const std::vector<std::string>& target_fields,
    size_t bucket_size, bool shuffle)
    : bucket_size_(bucket_size), shuffle_(shuffle) {
  provider_ = std::make_shared<CsvBatchProvider>(path, target_fields);

  ReadAndShuffle(0, false);
  if (!file_end_flag_) {
    ReadAndShuffle(1, true);
  }
}

std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>>
CachedCsvBatchProvider::ReadNextBatchWithIndex(size_t batch_size) {
  std::unique_lock lk(read_mutex_);

  std::vector<std::string> batch_data;
  std::vector<size_t> batch_indices;
  std::vector<size_t> shuffle_indices;

  if (file_end_flag_ && (bucket_items_[bucket_index_].size() == 0)) {
    SPDLOG_INFO("bucket_index_:{}, {}-{}", bucket_index_,
                bucket_items_[0].size(), bucket_items_[1].size());
    return std::make_tuple(batch_data, batch_indices, shuffle_indices);
  }
  SPU_ENFORCE(cursor_index_ <= bucket_items_[bucket_index_].size());

  size_t n_items =
      std::min(batch_size, bucket_items_[bucket_index_].size() - cursor_index_);

  for (size_t i = 0; i < n_items; ++i) {
    size_t shuffled_index = shuffled_indices_[bucket_index_][cursor_index_ + i];
    batch_data.push_back(bucket_items_[bucket_index_][shuffled_index]);
    batch_indices.push_back(bucket_count_ * bucket_size_ + cursor_index_ + i);
    shuffle_indices.push_back(bucket_count_ * bucket_size_ + shuffled_index);
  }

  cursor_index_ += n_items;

  if (cursor_index_ == bucket_items_[bucket_index_].size()) {
    SPDLOG_INFO("cursor_index_:{} n_items:{} batch_size:{}", cursor_index_,
                n_items, batch_size);
    std::unique_lock lk(bucket_mutex_[bucket_index_]);
    bucket_items_[bucket_index_].resize(0);
    cursor_index_ = 0;
  }

  if (n_items == batch_size) {
    return std::make_tuple(batch_data, batch_indices, shuffle_indices);
  }

  size_t next_index = 1 - bucket_index_;
  {
    // get next_index lock
    std::unique_lock lk(bucket_mutex_[next_index]);

    if (bucket_items_[next_index].size() > 0) {
      SPDLOG_INFO("lock idx:{}", next_index);

      size_t left_size = batch_size - n_items;

      cursor_index_ = 0;
      bucket_count_++;
      size_t m_items = std::min(left_size, bucket_items_[next_index].size());

      for (size_t i = 0; i < m_items; ++i) {
        size_t shuffled_index =
            shuffled_indices_[next_index][cursor_index_ + i];
        batch_data.push_back(bucket_items_[next_index][shuffled_index]);
        batch_indices.push_back(bucket_count_ * bucket_size_ + cursor_index_ +
                                i);
        shuffle_indices.push_back(bucket_count_ * bucket_size_ +
                                  shuffled_index);
      }

      if (m_items == bucket_items_[next_index].size()) {
        cursor_index_ = 0;
        bucket_items_[next_index].resize(0);
      } else {
        cursor_index_ += m_items;
      }
      n_items += m_items;

      // read
      SPDLOG_INFO("read next bucket, n_items:{} m_items:{}", n_items, m_items);
      if (!file_end_flag_) {
        ReadAndShuffle(bucket_index_, true);
      }

      bucket_index_ = next_index;

      SPDLOG_INFO("unlock idx:{}", next_index);
    }
  }

  return std::make_tuple(batch_data, batch_indices, shuffle_indices);
}

void CachedCsvBatchProvider::ReadAndShuffle(size_t read_index,
                                            bool thread_model) {
  SPDLOG_INFO("begin func for ReadAndShuffle read_index:{}", read_index);

  std::unique_lock<std::mutex> lk(bucket_mutex_[read_index]);

  auto read_proc = [&](int idx, std::unique_lock<std::mutex> lk) -> void {
    SPDLOG_INFO(
        "Begin thread ReadAndShuffle next bucket, read_index:{} "
        "bucket_size_:{}",
        idx, bucket_size_);

    SPDLOG_INFO("lock idx:{}", idx);
    {
      std::unique_lock<std::mutex> file_lk(file_mutex_);
      bucket_items_[idx] = provider_->ReadNextBatch(bucket_size_);
      if (bucket_items_[idx].empty() ||
          (bucket_items_[idx].size() < bucket_size_)) {
        file_end_flag_ = true;
      }

      shuffled_indices_[idx].resize(bucket_items_[idx].size());
      std::iota(shuffled_indices_[idx].begin(), shuffled_indices_[idx].end(),
                0);
    }

    if (shuffle_ && !bucket_items_[idx].empty()) {
      std::mt19937 rng(yacl::crypto::SecureRandU64());
      std::shuffle(shuffled_indices_[idx].begin(), shuffled_indices_[idx].end(),
                   rng);
    }
    SPDLOG_INFO("unlock idx:{}", idx);

    SPDLOG_INFO("End thread ReadAndShuffle next bucket[{}] {}", idx,
                bucket_items_[idx].size());
  };

  f_read_[read_index] =
      std::async(std::launch::async, read_proc, read_index, std::move(lk));
  if (!thread_model) {
    f_read_[read_index].get();
  }
  SPDLOG_INFO("end func ReadAndShuffle read_index:{}", read_index);
}

}  // namespace spu::psi
