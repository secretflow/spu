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

std::vector<std::string> MemoryBatchProvider::ReadNextBatch(size_t batch_size) {
  std::vector<std::string> batch;
  SPU_ENFORCE(cursor_index_ <= items_.size());
  size_t n_items = std::min(batch_size, items_.size() - cursor_index_);
  batch.insert(batch.end(), items_.begin() + cursor_index_,
               items_.begin() + cursor_index_ + n_items);
  cursor_index_ += n_items;
  return batch;
}

const std::vector<std::string>& MemoryBatchProvider::items() const {
  return items_;
}

CsvBatchProvider::CsvBatchProvider(
    const std::string& path, const std::vector<std::string>& target_fields)
    : path_(path), analyzer_(path, target_fields) {
  in_ = io::BuildInputStream(io::FileIoOptions(path_));
  // skip header
  std::string line;
  in_->GetLine(&line);
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

CachedCsvBatchProvider::CachedCsvBatchProvider(
    const std::string& path, const std::vector<std::string>& target_fields,
    size_t bucket_size, bool shuffle)
    : bucket_size_(bucket_size), shuffle_(shuffle) {
  provider_ = std::make_shared<CsvBatchProvider>(path, target_fields);

  ReadAndShuffle(0, false);
  ReadAndShuffle(1, true);
}

std::vector<std::string> CachedCsvBatchProvider::ReadNextBatch(
    size_t batch_size) {
  std::vector<std::string> batch;

  SPU_ENFORCE(cursor_index_ <= bucket_items_[bucket_index_].size());

  size_t n_items =
      std::min(batch_size, bucket_items_[bucket_index_].size() - cursor_index_);
  batch.insert(batch.end(),
               bucket_items_[bucket_index_].begin() + cursor_index_,
               bucket_items_[bucket_index_].begin() + cursor_index_ + n_items);
  cursor_index_ += n_items;

  if (n_items < batch_size) {
    size_t next_index = (bucket_index_ + 1) % 2;
    if (!bucket_items_[next_index].empty()) {
      size_t left_size = batch_size - n_items;

      cursor_index_ = 0;
      size_t m_items =
          std::min(left_size, bucket_items_[next_index].size() - cursor_index_);
      batch.insert(batch.end(),
                   bucket_items_[next_index].begin() + cursor_index_,
                   bucket_items_[next_index].begin() + cursor_index_ + m_items);

      cursor_index_ += m_items;

      // read

      SPDLOG_INFO("read next bucket");
      ReadAndShuffle(bucket_index_, true);

      bucket_index_ = next_index;
    }
  }

  return batch;
}

void CachedCsvBatchProvider::ReadAndShuffle(size_t read_index,
                                            bool thread_model) {
  SPDLOG_INFO("begin func for ReadAndShuffle read_index:{}", read_index);

  auto read_proc = [&](int idx) -> void {
    SPDLOG_INFO(
        "Begin thread ReadAndShuffle next bucket, read_index:{} "
        "bucket_size_:{}",
        idx, bucket_size_);

    bucket_items_[idx] = provider_->ReadNextBatch(bucket_size_);

    if (shuffle_ && !bucket_items_[idx].empty()) {
      std::mt19937 rng(yacl::crypto::SecureRandU64());
      std::shuffle(bucket_items_[idx].begin(), bucket_items_[idx].end(), rng);
    }
    SPDLOG_INFO("End thread ReadAndShuffle next bucket[idx] {}", idx,
                bucket_items_[idx].size());
  };

  f_read_[read_index] = std::async(std::launch::async, read_proc, read_index);
  if (!thread_model) {
    f_read_[read_index].get();
  }
  SPDLOG_INFO("end func ReadAndShuffle read_index:{}", read_index);
}

}  // namespace spu::psi
