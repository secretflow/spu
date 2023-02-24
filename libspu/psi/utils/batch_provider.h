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

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "libspu/psi/io/io.h"
#include "libspu/psi/utils/csv_header_analyzer.h"

namespace spu::psi {

/// Interface which produce batch of strings.
class IBatchProvider {
 public:
  virtual ~IBatchProvider() = default;

  // Read at most `batch_size` items and return them. An empty returned vector
  // is treated as the end of stream.
  virtual std::vector<std::string> ReadNextBatch(size_t batch_size) = 0;
};

class IShuffleBatchProvider {
 public:
  virtual ~IShuffleBatchProvider() = default;

  // Read at most `batch_size` items and return data and shuffle index.
  // An empty returned vector is treated as the end of stream.
  virtual std::tuple<std::vector<std::string>, std::vector<size_t>,
                     std::vector<size_t>>
  ReadNextBatchWithIndex(size_t batch_size) = 0;
};

class MemoryBatchProvider : public IBatchProvider,
                            public IShuffleBatchProvider {
 public:
  explicit MemoryBatchProvider(const std::vector<std::string>& items,
                               bool shuffle = false);
  std::vector<std::string> ReadNextBatch(size_t batch_size) override;

  std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>>
  ReadNextBatchWithIndex(size_t batch_size) override;

  const std::vector<std::string>& items() const;

  const std::vector<size_t>& shuffled_indices() const;

 private:
  const std::vector<std::string>& items_;
  std::vector<size_t> shuffled_indices_;
  bool shuffle_;
  size_t cursor_index_ = 0;
};

class CsvBatchProvider : public IBatchProvider {
 public:
  explicit CsvBatchProvider(const std::string& path,
                            const std::vector<std::string>& target_fields);

  std::vector<std::string> ReadNextBatch(size_t batch_size) override;

 private:
  const std::string path_;
  std::unique_ptr<io::InputStream> in_;
  CsvHeaderAnalyzer analyzer_;
};

class CachedCsvBatchProvider : public IShuffleBatchProvider {
 public:
  explicit CachedCsvBatchProvider(const std::string& path,
                                  const std::vector<std::string>& target_fields,
                                  size_t bucket_size = 100000000,
                                  bool shuffle = false);

  std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>>
  ReadNextBatchWithIndex(size_t batch_size) override;

 private:
  void ReadAndShuffle(size_t read_index, bool thread_model = false);

  std::shared_ptr<CsvBatchProvider> provider_;

  size_t bucket_size_;
  bool shuffle_;

  std::array<std::vector<std::string>, 2> bucket_items_;
  std::array<std::vector<size_t>, 2> shuffled_indices_;
  size_t cursor_index_ = 0;
  size_t bucket_index_ = 0;
  size_t bucket_count_ = 0;

  std::array<std::future<void>, 2> f_read_;

  std::array<std::mutex, 2> bucket_mutex_;
  std::mutex file_mutex_;
  bool file_end_flag_ = false;
};

}  // namespace spu::psi
