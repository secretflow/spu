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

#include <string>
#include <vector>

#include "yacl/base/buffer.h"

#include "libspu/psi/utils/serializable.pb.h"

namespace spu::psi::utils {

inline yacl::Buffer SerializeSize(size_t size) {
  proto::SizeProto proto;
  proto.set_input_size(size);
  yacl::Buffer buf(proto.ByteSizeLong());
  proto.SerializeToArray(buf.data(), buf.size());
  return buf;
}

inline size_t DeserializeSize(const yacl::Buffer& buf) {
  proto::SizeProto proto;
  proto.ParseFromArray(buf.data(), buf.size());
  return proto.input_size();
}

inline yacl::Buffer SerializeStrItems(const std::vector<std::string>& items) {
  proto::StrItemsProto proto;
  for (const auto& item : items) {
    proto.add_items(item);
  }
  yacl::Buffer buf(proto.ByteSizeLong());
  proto.SerializeToArray(buf.data(), buf.size());
  return buf;
}

inline void DeserializeStrItems(const yacl::Buffer& buf,
                                std::vector<std::string>* items) {
  proto::StrItemsProto proto;
  proto.ParseFromArray(buf.data(), buf.size());
  items->reserve(proto.items_size());
  for (auto item : proto.items()) {
    items->emplace_back(item);
  }
}

inline size_t GetCompareBytesLength(size_t size_a, size_t size_b,
                                    size_t stats_params = 40) {
  size_t compare_bits = std::ceil(std::log2(size_a)) +
                        std::ceil(std::log2(size_b)) + stats_params;

  return (compare_bits + 7) / 8;
}

}  // namespace spu::psi::utils
