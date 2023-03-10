// Copyright 2023 Ant Group Co., Ltd.
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
#include <tuple>
#include <utility>
#include <vector>

#include "yacl/base/byte_container_view.h"

#include "libspu/psi/io/io.h"
#include "libspu/psi/utils/batch_provider.h"

namespace spu::psi {

class UbPsiCacheProvider : public IBatchProvider, public IShuffleBatchProvider {
 public:
  UbPsiCacheProvider(const std::string &file_path, size_t data_len);
  ~UbPsiCacheProvider() { in_->Close(); }

  std::vector<std::string> ReadNextBatch(size_t batch_size) override;

  std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>>
  ReadNextBatchWithIndex(size_t batch_size) override;

  const std::vector<std::string> &GetSelectedFields();

 private:
  std::vector<std::tuple<std::string, size_t, size_t>> ReadData(
      size_t read_count);

  std::string file_path_;
  size_t file_size_;
  size_t file_cursor_ = 0;
  std::unique_ptr<io::InputStream> in_;
  size_t data_len_;
  size_t data_index_len_;

  std::vector<std::string> selected_fields_;
};

class IUbPsiCache {
 public:
  virtual ~IUbPsiCache() = default;

  virtual void SaveData(yacl::ByteContainerView item, size_t index,
                        size_t shuffle_index) = 0;

  virtual void Flush() { return; }
};

class UbPsiCache : public IUbPsiCache {
 public:
  UbPsiCache(const std::string &file_path, size_t data_len,
             const std::vector<std::string> &ids);

  ~UbPsiCache() { out_stream_->Close(); }

  void SaveData(yacl::ByteContainerView item, size_t index,
                size_t shuffle_index) override;

  void Flush() override { out_stream_->Flush(); }

 private:
  std::string file_path_;
  size_t data_len_;
  size_t data_index_len_;
  std::unique_ptr<io::OutputStream> out_stream_;
};

}  // namespace spu::psi
