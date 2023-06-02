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

#include "libspu/pir/pir.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "yacl/crypto/utils/rand.h"
#include "yacl/io/kv/leveldb_kvstore.h"
#include "yacl/io/rw/csv_writer.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/core/labeled_psi/sender.h"
#include "libspu/psi/core/labeled_psi/sender_db.h"
#include "libspu/psi/cryptor/ecc_cryptor.h"
#include "libspu/psi/io/io.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

namespace spu::pir {

namespace {

std::vector<uint8_t> ReadEcSecretKeyFile(const std::string &file_path) {
  size_t file_byte_size = 0;
  try {
    file_byte_size = std::filesystem::file_size(file_path);
  } catch (std::filesystem::filesystem_error &e) {
    SPU_THROW("ReadEcSecretKeyFile {} Error: {}", file_path, e.what());
  }
  SPU_ENFORCE(file_byte_size == spu::psi::kEccKeySize,
              "error format: key file bytes is not {}", spu::psi::kEccKeySize);

  std::vector<uint8_t> secret_key(spu::psi::kEccKeySize);

  auto in =
      spu::psi::io::BuildInputStream(spu::psi::io::FileIoOptions(file_path));
  in->Read(secret_key.data(), spu::psi::kEccKeySize);
  in->Close();

  return secret_key;
}

size_t CsvFileDataCount(const std::string &file_path,
                        const std::vector<std::string> &ids) {
  size_t data_count = 0;

  std::shared_ptr<spu::psi::IBatchProvider> batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(file_path, ids);

  while (true) {
    auto batch = batch_provider->ReadNextBatch(4096);
    if (batch.empty()) {
      break;
    }
    data_count += batch.size();
  }

  return data_count;
}

constexpr char kMetaInfoStoreName[] = "meta_info";
constexpr char kServerDataCount[] = "server_data_count";
constexpr char kCountPerQuery[] = "count_per_query";
constexpr char kLabelByteCount[] = "label_byte_count";
constexpr char kLabelColumns[] = "label_columns";
constexpr char kPsiParams[] = "psi_params";

void WriteMetaInfo(const std::string &setup_path, size_t server_data_count,
                   size_t count_per_query, size_t label_byte_count,
                   const std::vector<std::string> &label_cloumns,
                   const apsi::PSIParams &psi_params) {
  std::string meta_store_name =
      fmt::format("{}/{}", setup_path, kMetaInfoStoreName);

  std::shared_ptr<yacl::io::KVStore> meta_info_store =
      std::make_shared<yacl::io::LeveldbKVStore>(false, meta_store_name);

  meta_info_store->Put(kServerDataCount, fmt::format("{}", server_data_count));
  meta_info_store->Put(kCountPerQuery, fmt::format("{}", count_per_query));
  meta_info_store->Put(kLabelByteCount, fmt::format("{}", label_byte_count));

  spu::psi::proto::StrItemsProto proto;
  for (const auto &label_cloumn : label_cloumns) {
    proto.add_items(label_cloumn);
  }
  yacl::Buffer buf(proto.ByteSizeLong());
  proto.SerializeToArray(buf.data(), buf.size());

  meta_info_store->Put(kLabelColumns, buf);

  yacl::Buffer params_buffer = spu::psi::PsiParamsToBuffer(psi_params);
  meta_info_store->Put(kPsiParams, params_buffer);
}

size_t GetSizeFromStore(
    const std::shared_ptr<yacl::io::KVStore> &meta_info_store,
    const std::string &key_name) {
  yacl::Buffer temp_value;
  meta_info_store->Get(key_name, &temp_value);
  size_t key_value = std::stoul(std::string(std::string_view(
      reinterpret_cast<char *>(temp_value.data()), temp_value.size())));

  return key_value;
}

apsi::PSIParams ReadMetaInfo(const std::string &setup_path,
                             size_t *server_data_count, size_t *count_per_query,
                             size_t *label_byte_count,
                             std::vector<std::string> *label_cloumns) {
  std::string meta_store_name =
      fmt::format("{}/{}", setup_path, kMetaInfoStoreName);
  std::shared_ptr<yacl::io::KVStore> meta_info_store =
      std::make_shared<yacl::io::LeveldbKVStore>(false, meta_store_name);

  *server_data_count = GetSizeFromStore(meta_info_store, kServerDataCount);
  *count_per_query = GetSizeFromStore(meta_info_store, kCountPerQuery);
  *label_byte_count = GetSizeFromStore(meta_info_store, kLabelByteCount);

  yacl::Buffer label_columns_buf;
  meta_info_store->Get(kLabelColumns, &label_columns_buf);
  spu::psi::proto::StrItemsProto proto;
  proto.ParseFromArray(label_columns_buf.data(), label_columns_buf.size());
  (*label_cloumns).reserve(proto.items_size());
  for (auto item : proto.items()) {
    (*label_cloumns).emplace_back(item);
  }

  yacl::Buffer params_buffer;
  meta_info_store->Get(kPsiParams, &params_buffer);
  apsi::PSIParams psi_params = spu::psi::ParsePsiParamsProto(params_buffer);

  return psi_params;
}

}  // namespace

PirResultReport LabeledPirSetup(const PirSetupConfig &config) {
  std::vector<std::string> key_columns;
  key_columns.insert(key_columns.end(), config.key_columns().begin(),
                     config.key_columns().end());

  std::vector<std::string> label_columns;
  label_columns.insert(label_columns.end(), config.label_columns().begin(),
                       config.label_columns().end());

  size_t server_data_count = CsvFileDataCount(config.input_path(), key_columns);
  size_t count_per_query = config.num_per_query();

  apsi::PSIParams psi_params =
      spu::psi::GetPsiParams(count_per_query, server_data_count);

  // spu::pir::examples::utils::WritePsiParams(ParamsOutPathOpt.getValue(),
  //                                           psi_params);

  std::vector<uint8_t> oprf_key = ReadEcSecretKeyFile(config.oprf_key_path());

  size_t label_byte_count = config.label_max_len();
  size_t nonce_byte_count = 16;

  std::string kv_store_path = config.setup_path();
  // delete store path
  {
    std::error_code ec;
    std::filesystem::remove_all(kv_store_path, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp dir: {}, msg: {}", kv_store_path,
                  ec.message());
    }
  }
  std::filesystem::create_directory(kv_store_path);

  WriteMetaInfo(kv_store_path, server_data_count, count_per_query,
                label_byte_count, label_columns, psi_params);

  std::shared_ptr<spu::psi::SenderDB> sender_db =
      std::make_shared<spu::psi::SenderDB>(psi_params, oprf_key, kv_store_path,
                                           label_byte_count, nonce_byte_count,
                                           false);

  std::shared_ptr<spu::psi::IBatchProvider> batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(config.input_path(),
                                                   key_columns, label_columns);

  sender_db->SetData(batch_provider);

  PirResultReport report;
  report.set_data_count(server_data_count);

  return report;
}

PirResultReport LabeledPirServer(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const std::shared_ptr<spu::psi::SenderDB> &sender_db,
    const std::vector<uint8_t> &oprf_key, const apsi::PSIParams &psi_params,
    const std::vector<std::string> &label_columns, size_t server_data_count,
    size_t count_per_query, size_t label_byte_count) {
  // send count_per_query
  link_ctx->SendAsync(link_ctx->NextRank(),
                      spu::psi::utils::SerializeSize(count_per_query),
                      fmt::format("count_per_query:{}", count_per_query));

  yacl::Buffer labels_buffer =
      spu::psi::utils::SerializeStrItems(label_columns);
  // send labels column name
  link_ctx->SendAsync(link_ctx->NextRank(), labels_buffer,
                      fmt::format("send label columns name"));

  // send psi params
  yacl::Buffer params_buffer = spu::psi::PsiParamsToBuffer(psi_params);
  link_ctx->SendAsync(link_ctx->NextRank(), params_buffer,
                      fmt::format("send psi params"));

  // const auto total_query_start = std::chrono::system_clock::now();

  size_t query_count = 0;
  size_t data_count = 0;

  spu::psi::LabelPsiSender sender(sender_db);

  SPDLOG_INFO("LabelPsiSender");

  while (true) {
    // recv current batch_size
    size_t batch_data_size = spu::psi::utils::DeserializeSize(
        link_ctx->Recv(link_ctx->NextRank(), fmt::format("batch_data_size")));

    SPDLOG_INFO("client data size: {}", batch_data_size);
    if (batch_data_size == 0) {
      break;
    }
    data_count += batch_data_size;

    // oprf
    std::unique_ptr<spu::psi::IEcdhOprfServer> oprf_server =
        spu::psi::CreateEcdhOprfServer(oprf_key, spu::psi::OprfType::Basic,
                                       spu::psi::CurveType::CURVE_FOURQ);

    // const auto oprf_start = std::chrono::system_clock::now();
    sender.RunOPRF(std::move(oprf_server), link_ctx);

    // const auto oprf_end = std::chrono::system_clock::now();
    // const DurationMillis oprf_duration = oprf_end - oprf_start;
    // SPDLOG_INFO("*** server oprf duration:{}", oprf_duration.count());

    // const auto query_start = std::chrono::system_clock::now();

    sender.RunQuery(link_ctx);

    // const auto query_end = std::chrono::system_clock::now();
    // const DurationMillis query_duration = query_end - query_start;
    // SPDLOG_INFO("*** server query duration:{}", query_duration.count());

    query_count++;
  }
  SPDLOG_INFO("query_count:{},data_count:{}", query_count, data_count);

  PirResultReport report;
  report.set_data_count(data_count);

  return report;
}

PirResultReport LabeledPirServer(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirServerConfig &config) {
  size_t server_data_count;
  size_t count_per_query;
  size_t label_byte_count;

  std::vector<std::string> label_columns;

  apsi::PSIParams psi_params =
      ReadMetaInfo(config.setup_path(), &server_data_count, &count_per_query,
                   &label_byte_count, &label_columns);

  SPU_ENFORCE(label_columns.size() > 0);

  std::vector<uint8_t> oprf_key = ReadEcSecretKeyFile(config.oprf_key_path());

  SPDLOG_INFO("table_params hash_func_count:{}",
              psi_params.table_params().hash_func_count);

  size_t nonce_byte_count = 16;

  bool compressed = false;
  std::shared_ptr<spu::psi::SenderDB> sender_db =
      std::make_shared<spu::psi::SenderDB>(
          psi_params, oprf_key, config.setup_path(), label_byte_count,
          nonce_byte_count, compressed);

  SPDLOG_INFO("db GetItemCount:{}", sender_db->GetItemCount());

  // send count_per_query
  link_ctx->SendAsync(link_ctx->NextRank(),
                      spu::psi::utils::SerializeSize(count_per_query),
                      fmt::format("count_per_query:{}", count_per_query));

  yacl::Buffer labels_buffer =
      spu::psi::utils::SerializeStrItems(label_columns);
  // send labels column name
  link_ctx->SendAsync(link_ctx->NextRank(), labels_buffer,
                      fmt::format("send label columns name"));

  // send psi params
  yacl::Buffer params_buffer = spu::psi::PsiParamsToBuffer(psi_params);
  link_ctx->SendAsync(link_ctx->NextRank(), params_buffer,
                      fmt::format("send psi params"));

  // const auto total_query_start = std::chrono::system_clock::now();

  size_t query_count = 0;
  size_t data_count = 0;

  while (true) {
    // recv current batch_size
    size_t batch_data_size = spu::psi::utils::DeserializeSize(
        link_ctx->Recv(link_ctx->NextRank(), fmt::format("batch_data_size")));

    SPDLOG_INFO("client data size: {}", batch_data_size);
    if (batch_data_size == 0) {
      break;
    }
    data_count += batch_data_size;

    // oprf
    std::unique_ptr<spu::psi::IEcdhOprfServer> oprf_server =
        spu::psi::CreateEcdhOprfServer(oprf_key, spu::psi::OprfType::Basic,
                                       spu::psi::CurveType::CURVE_FOURQ);

    spu::psi::LabelPsiSender sender(sender_db);

    // const auto oprf_start = std::chrono::system_clock::now();
    sender.RunOPRF(std::move(oprf_server), link_ctx);

    // const auto oprf_end = std::chrono::system_clock::now();
    // const DurationMillis oprf_duration = oprf_end - oprf_start;
    // SPDLOG_INFO("*** server oprf duration:{}", oprf_duration.count());

    // const auto query_start = std::chrono::system_clock::now();

    sender.RunQuery(link_ctx);

    // const auto query_end = std::chrono::system_clock::now();
    // const DurationMillis query_duration = query_end - query_start;
    // SPDLOG_INFO("*** server query duration:{}", query_duration.count());

    query_count++;
  }
  SPDLOG_INFO("query_count:{},data_count:{}", query_count, data_count);

  PirResultReport report;
  report.set_data_count(data_count);

  return report;
}

PirResultReport LabeledPirMemoryServer(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirSetupConfig &config) {
  std::vector<std::string> key_columns;
  key_columns.insert(key_columns.end(), config.key_columns().begin(),
                     config.key_columns().end());

  std::vector<std::string> label_columns;
  label_columns.insert(label_columns.end(), config.label_columns().begin(),
                       config.label_columns().end());

  size_t server_data_count = CsvFileDataCount(config.input_path(), key_columns);
  size_t count_per_query = config.num_per_query();
  SPDLOG_INFO("server_data_count:{}", server_data_count);

  apsi::PSIParams psi_params =
      spu::psi::GetPsiParams(count_per_query, server_data_count);

  std::vector<uint8_t> oprf_key =
      yacl::crypto::RandBytes(spu::psi::kEccKeySize);

  size_t label_byte_count = config.label_max_len();
  size_t nonce_byte_count = 16;

  std::shared_ptr<spu::psi::SenderDB> sender_db =
      std::make_shared<spu::psi::SenderDB>(psi_params, oprf_key, "::memory",
                                           label_byte_count, nonce_byte_count,
                                           false);

  std::shared_ptr<spu::psi::IBatchProvider> batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(config.input_path(),
                                                   key_columns, label_columns);

  sender_db->SetData(batch_provider);

  SPDLOG_INFO("sender_db->GetItemCount:{}", sender_db->GetItemCount());

  PirResultReport report = LabeledPirServer(
      link_ctx, sender_db, oprf_key, psi_params, label_columns,
      sender_db->GetItemCount(), count_per_query, label_byte_count);

  return report;
}

PirResultReport LabeledPirClient(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirClientConfig &config) {
  std::vector<std::string> key_columns;
  key_columns.insert(key_columns.end(), config.key_columns().begin(),
                     config.key_columns().end());

  // recv count_per_query
  size_t count_per_query = spu::psi::utils::DeserializeSize(
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("count_per_query")));

  SPU_ENFORCE(count_per_query > 0, "Invalid nr:{}", count_per_query);

  // recv label columns
  yacl::Buffer label_columns_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv label columns name"));

  std::vector<std::string> label_columns_name;
  spu::psi::utils::DeserializeStrItems(label_columns_buffer,
                                       &label_columns_name);

  std::shared_ptr<spu::psi::IBatchProvider> query_batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(config.input_path(),
                                                   key_columns);

  yacl::io::Schema s;
  for (size_t i = 0; i < key_columns.size(); ++i) {
    s.feature_types.push_back(yacl::io::Schema::STRING);
  }
  for (size_t i = 0; i < label_columns_name.size(); ++i) {
    s.feature_types.push_back(yacl::io::Schema::STRING);
  }

  s.feature_names = key_columns;
  s.feature_names.insert(s.feature_names.end(), label_columns_name.begin(),
                         label_columns_name.end());

  yacl::io::WriterOptions w_op;
  w_op.file_schema = s;

  auto out = spu::psi::io::BuildOutputStream(
      spu::psi::io::FileIoOptions(config.output_path()));
  yacl::io::CsvWriter writer(w_op, std::move(out));
  writer.Init();

  // recv psi params
  yacl::Buffer params_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv psi params"));

  apsi::PSIParams psi_params = spu::psi::ParsePsiParamsProto(params_buffer);

  spu::psi::LabelPsiReceiver receiver(psi_params, true);

  // const auto total_query_start = std::chrono::system_clock::now();

  size_t query_count = 0;
  size_t data_count = 0;

  while (true) {
    auto query_batch_items =
        query_batch_provider->ReadNextBatch(count_per_query);

    // send count_batch_size
    link_ctx->SendAsync(
        link_ctx->NextRank(),
        spu::psi::utils::SerializeSize(query_batch_items.size()),
        fmt::format("count_batch_size:{}", query_batch_items.size()));

    if (query_batch_items.empty()) {
      break;
    }
    data_count += query_batch_items.size();

    // const auto oprf_start = std::chrono::system_clock::now();
    std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>
        items_oprf = receiver.RequestOPRF(query_batch_items, link_ctx);

    // const auto oprf_end = std::chrono::system_clock::now();
    // const DurationMillis oprf_duration = oprf_end - oprf_start;
    // SPDLOG_INFO("*** server oprf duration:{}", oprf_duration.count());

    // const auto query_start = std::chrono::system_clock::now();
    std::pair<std::vector<size_t>, std::vector<std::string>> query_result =
        receiver.RequestQuery(items_oprf.first, items_oprf.second, link_ctx);

    // const auto query_end = std::chrono::system_clock::now();
    // const DurationMillis query_duration = query_end - query_start;
    // SPDLOG_INFO("*** server query duration:{}", query_duration.count());

    SPDLOG_INFO("query_result size:{}", query_result.first.size());

    yacl::io::ColumnVectorBatch batch;

    std::vector<std::vector<std::string>> query_id_results(key_columns.size());
    std::vector<std::vector<std::string>> query_label_results(
        label_columns_name.size());

    for (size_t i = 0; i < query_result.first.size(); ++i) {
      std::vector<std::string> result_ids =
          absl::StrSplit(query_batch_items[query_result.first[i]], ",");

      SPU_ENFORCE(result_ids.size() == key_columns.size());

      std::vector<std::string> result_labels =
          absl::StrSplit(query_result.second[i], ",");
      SPU_ENFORCE(result_labels.size() == label_columns_name.size());

      for (size_t j = 0; j < result_ids.size(); ++j) {
        query_id_results[j].push_back(result_ids[j]);
      }
      for (size_t j = 0; j < result_labels.size(); ++j) {
        query_label_results[j].push_back(result_labels[j]);
      }
    }

    for (size_t i = 0; i < key_columns.size(); ++i) {
      batch.AppendCol(query_id_results[i]);
    }
    for (size_t i = 0; i < label_columns_name.size(); ++i) {
      batch.AppendCol(query_label_results[i]);
    }

    writer.Add(batch);

    query_count++;
  }

  writer.Close();
  SPDLOG_INFO("query_count:{}, data_count:{}", query_count, data_count);

  PirResultReport report;
  report.set_data_count(data_count);

  return report;
}

PirResultReport PirSetup(const PirSetupConfig &config) {
  if (config.pir_protocol() != KEYWORD_PIR_LABELED_PSI) {
    SPU_THROW("Unsupported pir protocol {}", config.pir_protocol());
  }

  return LabeledPirSetup(config);
}

PirResultReport PirServer(const std::shared_ptr<yacl::link::Context> &link_ctx,
                          const PirServerConfig &config) {
  if (config.pir_protocol() != KEYWORD_PIR_LABELED_PSI) {
    SPU_THROW("Unsupported pir protocol {}", config.pir_protocol());
  }

  return LabeledPirServer(link_ctx, config);
}

PirResultReport PirMemoryServer(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirSetupConfig &config) {
  if (config.pir_protocol() != KEYWORD_PIR_LABELED_PSI) {
    SPU_THROW("Unsupported pir protocol {}", config.pir_protocol());
  }

  return LabeledPirMemoryServer(link_ctx, config);
}

PirResultReport PirClient(const std::shared_ptr<yacl::link::Context> &link_ctx,
                          const PirClientConfig &config) {
  if (config.pir_protocol() != KEYWORD_PIR_LABELED_PSI) {
    SPU_THROW("Unsupported pir protocol {}", config.pir_protocol());
  }

  return LabeledPirClient(link_ctx, config);
}

}  // namespace spu::pir
