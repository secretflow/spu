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

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "apsi/util/db_encoding.h"
#include "spdlog/spdlog.h"

#include "libspu/psi/core/labeled_psi/serializable.pb.h"

namespace spu::psi {

inline void SerializeAlgItem(const apsi::util::AlgItem& alg_items,
                             proto::AlgItemProto* proto) {
  for (auto& alg_item : alg_items) {
    proto->add_item(alg_item);
  }
}

inline proto::AlgItemProto SerializeAlgItem(
    const apsi::util::AlgItem& alg_items) {
  proto::AlgItemProto proto;

  SerializeAlgItem(alg_items, &proto);

  return proto;
}

inline std::string SerializeAlgItemToString(
    const apsi::util::AlgItem& alg_items) {
  std::string ret;

  proto::AlgItemProto proto = SerializeAlgItem(alg_items);

  std::string item_string(proto.ByteSizeLong(), '\0');
  proto.SerializePartialToArray(item_string.data(), item_string.length());

  return item_string;
}

inline apsi::util::AlgItem DeserializeAlgItem(
    const proto::AlgItemProto& proto) {
  apsi::util::AlgItem alg_items;

  alg_items.resize(proto.item_size());
  for (int i = 0; i < proto.item_size(); ++i) {
    alg_items[i] = proto.item(i);
  }

  return alg_items;
}

inline apsi::util::AlgItem DeserializeAlgItem(const absl::string_view& buf) {
  apsi::util::AlgItem alg_items;
  proto::AlgItemProto proto;
  proto.ParseFromArray(buf.data(), buf.length());

  return DeserializeAlgItem(proto);
}

inline void SerializeAlgItemLabel(
    const apsi::util::AlgItemLabel& item_label_pair,
    proto::AlgItemLabelProto* proto) {
  for (size_t i = 0; i < item_label_pair.size(); ++i) {
    proto::AlgItemLabelPairProto* pair_proto = proto->add_item_label();

    pair_proto->set_item(item_label_pair[i].first);
    for (auto& label : item_label_pair[i].second) {
      pair_proto->add_label(label);
    }
  }
}

inline proto::AlgItemLabelProto SerializeAlgItemLabel(
    const apsi::util::AlgItemLabel& item_label_pair) {
  proto::AlgItemLabelProto proto;
  SerializeAlgItemLabel(item_label_pair, &proto);

  return proto;
}

inline std::string SerializeAlgItemLabelToString(
    const apsi::util::AlgItemLabel& item_label_pair) {
  proto::AlgItemLabelProto proto = SerializeAlgItemLabel(item_label_pair);

  std::string item_string(proto.ByteSizeLong(), '\0');
  proto.SerializePartialToArray(item_string.data(), item_string.length());

  return item_string;
}

inline apsi::util::AlgItemLabel DeserializeAlgItemLabel(
    const proto::AlgItemLabelProto& proto) {
  apsi::util::AlgItemLabel item_label_pair;

  for (int i = 0; i < proto.item_label_size(); ++i) {
    auto pair_proto = proto.item_label(i);
    std::vector<apsi::util::felt_t> labels(pair_proto.label_size());
    for (int j = 0; j < pair_proto.label_size(); ++j) {
      labels[j] = pair_proto.label(j);
    }
    item_label_pair.emplace_back(pair_proto.item(), labels);
  }

  return item_label_pair;
}

inline apsi::util::AlgItemLabel DeserializeAlgItemLabel(
    const absl::string_view& buf) {
  proto::AlgItemLabelProto proto;
  proto.ParseFromArray(buf.data(), buf.size());

  return DeserializeAlgItemLabel(proto);
}

inline std::string SerializeDataWithIndices(
    const std::pair<apsi::util::AlgItem, size_t>& data_with_indices) {
  proto::DataWithIndicesProto proto;

  proto::AlgItemProto* item_proto = new proto::AlgItemProto();
  SerializeAlgItem(data_with_indices.first, item_proto);

  proto.set_allocated_data(item_proto);
  proto.set_index(data_with_indices.second);

  std::string item_string(proto.ByteSizeLong(), '\0');
  proto.SerializePartialToArray(item_string.data(), proto.ByteSizeLong());

  // SPDLOG_INFO("*** debug item_size:{} bytes:{} proto size:{}",
  //             item_proto->item_size(), item_proto->ByteSizeLong(),
  //             proto.ByteSizeLong());

  return item_string;
}

inline std::pair<apsi::util::AlgItem, size_t> DeserializeDataWithIndices(
    const absl::string_view& buf) {
  proto::DataWithIndicesProto proto;
  proto.ParseFromArray(buf.data(), buf.size());

  apsi::util::AlgItem alg_item = DeserializeAlgItem(proto.data());

  return std::make_pair(alg_item, proto.index());
}

inline std::string SerializeDataLabelWithIndices(
    const std::pair<apsi::util::AlgItemLabel, size_t>& data_with_indices) {
  std::string ret;

  proto::DataLabelWithIndicesProto proto;

  proto::AlgItemLabelProto* item_proto = new proto::AlgItemLabelProto();
  SerializeAlgItemLabel(data_with_indices.first, item_proto);

  proto.set_allocated_data(item_proto);
  proto.set_index(data_with_indices.second);

  std::string item_string(proto.ByteSizeLong(), '\0');
  proto.SerializePartialToArray(item_string.data(), item_string.length());

  return item_string;
}

inline std::pair<apsi::util::AlgItemLabel, size_t>
DeserializeDataLabelWithIndices(const absl::string_view& buf) {
  proto::DataLabelWithIndicesProto proto;
  proto.ParseFromArray(buf.data(), buf.size());

  apsi::util::AlgItemLabel alg_item = DeserializeAlgItemLabel(proto.data());

  return std::make_pair(alg_item, proto.index());
}

}  // namespace spu::psi
