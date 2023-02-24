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

std::tuple<std::string, size_t, size_t> UbPsiCacheProvider::ReadData() {
  std::vector<uint8_t> read_data(data_index_len_);
  std::string item(data_len_, '\0');
  uint64_t index, shuffle_index;

  in_->Read(read_data.data(), data_index_len_);

  std::memcpy(item.data(), read_data.data(), data_len_);
  std::memcpy(&index, read_data.data() + data_len_, sizeof(uint64_t));
  std::memcpy(&shuffle_index, read_data.data() + data_len_ + sizeof(uint64_t),
              sizeof(uint64_t));

  return std::make_tuple(item, index, shuffle_index);
}

std::vector<std::string> UbPsiCacheProvider::ReadNextBatch(size_t batch_size) {
  std::vector<std::string> ret;

  while ((file_cursor_ != file_size_) && !in_->Eof()) {
    std::string item;
    size_t index, shuffle_index;

    std::tie(item, index, shuffle_index) = ReadData();
    file_cursor_ += data_index_len_;

    ret.push_back(item);

    if ((ret.size() == batch_size) || (file_cursor_ == file_size_)) {
      break;
    }
  }

  return ret;
}

std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>>
UbPsiCacheProvider::ReadNextBatchWithIndex(size_t batch_size) {
  std::vector<std::string> ret_data;
  std::vector<size_t> ret_indices;
  std::vector<size_t> shuffle_indices;

  while ((file_cursor_ != file_size_) && !in_->Eof()) {
    std::string item;
    size_t index, shuffle_index;
    std::tie(item, index, shuffle_index) = ReadData();
    file_cursor_ += data_index_len_;

    ret_data.push_back(item);
    ret_indices.push_back(index);
    shuffle_indices.push_back(shuffle_index);

    if ((ret_data.size() == batch_size) || (file_cursor_ == file_size_)) {
      break;
    }
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
