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

#include "libspu/psi/utils/ub_psi_cache.h"

#include <algorithm>
#include <tuple>

#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/serialize.h"

namespace spu::psi {

UbPsiCacheProvider::UbPsiCacheProvider(const std::string &file_path,
                                       size_t data_len)
    : file_path_(file_path), data_len_(data_len) {
  in_ = io::BuildInputStream(io::FileIoOptions(file_path));
  file_size_ = in_->GetLength();

  data_index_len_ = data_len_ + 2 * sizeof(uint64_t);

  uint64_t ids_buffer_len;
  in_->Read(&ids_buffer_len, sizeof(uint64_t));

  file_cursor_ += sizeof(uint64_t);

  if (ids_buffer_len > 0) {
    yacl::Buffer ids_buffer(ids_buffer_len);
    in_->Read(ids_buffer.data(), ids_buffer_len);
    utils::DeserializeStrItems(ids_buffer, &selected_fields_);
    file_cursor_ += ids_buffer_len;
  }
}

std::vector<std::tuple<std::string, size_t, size_t>>
UbPsiCacheProvider::ReadData(size_t read_count) {
  std::vector<uint8_t> read_data(read_count * data_index_len_);
  std::string item(data_len_, '\0');
  size_t index, shuffle_index;

  std::vector<std::tuple<std::string, size_t, size_t>> ret;

  in_->Read(read_data.data(), read_count * data_index_len_);
  for (size_t i = 0; i < read_count; ++i) {
    size_t cur_pos = i * data_index_len_;
    std::memcpy(item.data(), read_data.data() + cur_pos, data_len_);
    std::memcpy(&index, read_data.data() + cur_pos + data_len_, sizeof(size_t));
    std::memcpy(&shuffle_index,
                read_data.data() + cur_pos + data_len_ + sizeof(size_t),
                sizeof(size_t));
    ret.push_back(std::make_tuple(item, index, shuffle_index));
  }

  return ret;
}

std::vector<std::string> UbPsiCacheProvider::ReadNextBatch(size_t batch_size) {
  std::vector<std::string> ret;

  size_t read_bytes =
      std::min<size_t>(batch_size * data_index_len_, file_size_ - file_cursor_);
  size_t read_count = read_bytes / data_index_len_;

  if (read_bytes > 0) {
    std::vector<std::tuple<std::string, size_t, size_t>> data =
        ReadData(read_count);

    for (const auto &d : data) {
      ret.push_back(std::get<0>(d));
    }
    file_cursor_ += read_bytes;
  }

  return ret;
}

std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>>
UbPsiCacheProvider::ReadNextBatchWithIndex(size_t batch_size) {
  std::vector<std::string> ret_data;
  std::vector<size_t> ret_indices;
  std::vector<size_t> shuffle_indices;

  size_t read_bytes =
      std::min<size_t>(batch_size * data_index_len_, file_size_ - file_cursor_);
  size_t read_count = read_bytes / data_index_len_;

  if (read_bytes > 0) {
    std::vector<std::tuple<std::string, size_t, size_t>> data =
        ReadData(read_count);
    for (const auto &d : data) {
      ret_data.push_back(std::get<0>(d));
      ret_indices.push_back(std::get<1>(d));
      shuffle_indices.push_back(std::get<2>(d));
    }
    file_cursor_ += read_bytes;
  }

  return std::make_tuple(ret_data, ret_indices, shuffle_indices);
}

const std::vector<std::string> &UbPsiCacheProvider::GetSelectedFields() {
  return selected_fields_;
}

UbPsiCache::UbPsiCache(const std::string &file_path, size_t data_len,
                       const std::vector<std::string> &selected_fields)
    : file_path_(file_path), data_len_(data_len) {
  out_stream_ = io::BuildOutputStream(io::FileIoOptions(file_path));
  data_index_len_ = data_len_ + 2 * sizeof(uint64_t);

  yacl::Buffer ids_buffer = utils::SerializeStrItems(selected_fields);
  uint64_t buffer_len = ids_buffer.size();

  out_stream_->Write(&buffer_len, sizeof(uint64_t));
  if (buffer_len > 0) {
    out_stream_->Write(ids_buffer.data(), ids_buffer.size());
  }
}

void UbPsiCache::SaveData(yacl::ByteContainerView item, size_t index,
                          size_t shuffle_index) {
  SPU_ENFORCE(item.size() == data_len_, "item size:{} data_len_:{}",
              item.size(), data_len_);
  std::string data_with_index(data_index_len_, '\0');

  std::memcpy(data_with_index.data(), item.data(), data_len_);
  std::memcpy(data_with_index.data() + data_len_, &index, sizeof(uint64_t));
  std::memcpy(data_with_index.data() + data_len_ + sizeof(uint64_t),
              &shuffle_index, sizeof(uint64_t));

  out_stream_->Write(data_with_index.data(), data_with_index.length());
}

}  // namespace spu::psi
