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

// clang-format off
// build keyword_pir_client
// > bazel build //examples/cpp/pir:keyword_pir_client -c opt
//
// To run the example, start terminals:
// > ./keyword_pir_client -rank 1 -in_path ../../data/psi_client_data.csv.csv 
//        -key_columns id -data_per_query 256 -out_path pir_out.csv
// clang-format on

#include <chrono>
#include <filesystem>
#include <string>

#include "examples/cpp/pir/keyword_pir_utils.h"
#include "examples/cpp/utils.h"
#include "yacl/io/rw/csv_writer.h"

#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> InPathOpt("in_path", llvm::cl::init("data.csv"),
                                     llvm::cl::desc("pir data in file path"));

llvm::cl::opt<std::string> KeyColumnsOpt("key_columns", llvm::cl::init("id"),
                                         llvm::cl::desc("key columns"));

llvm::cl::opt<int> DataPerQueryOpt("data_per_query", llvm::cl::init(256),
                                   llvm::cl::desc("data count per query"));

llvm::cl::opt<std::string> OutPathOpt(
    "out_path", llvm::cl::init("."),
    llvm::cl::desc("[out] pir query output path for db setup data"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto hctx = MakeHalContext();
  auto link_ctx = hctx->lctx();

  std::vector<std::string> ids = absl::StrSplit(KeyColumnsOpt.getValue(), ',');

  // recv label columns
  yacl::Buffer label_columns_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv label columns name"));
  std::vector<std::string> label_columns_name;
  spu::psi::utils::DeserializeStrItems(label_columns_buffer,
                                       &label_columns_name);

  std::shared_ptr<spu::psi::IBatchProvider> query_batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(InPathOpt.getValue(), ids);

  yacl::io::Schema s;
  for (size_t i = 0; i < ids.size(); ++i) {
    s.feature_types.push_back(yacl::io::Schema::STRING);
  }
  for (size_t i = 0; i < label_columns_name.size(); ++i) {
    s.feature_types.push_back(yacl::io::Schema::STRING);
  }

  s.feature_names = ids;
  s.feature_names.insert(s.feature_names.end(), label_columns_name.begin(),
                         label_columns_name.end());

  yacl::io::WriterOptions w_op;
  w_op.file_schema = s;

  auto out = spu::psi::io::BuildOutputStream(
      spu::psi::io::FileIoOptions(OutPathOpt.getValue()));
  yacl::io::CsvWriter writer(w_op, std::move(out));
  writer.Init();

  size_t nr = DataPerQueryOpt.getValue();

  // recv psi params
  yacl::Buffer params_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv psi params"));

  apsi::PSIParams psi_params = spu::psi::ParsePsiParamsProto(params_buffer);

  spu::psi::LabelPsiReceiver receiver(psi_params, true);

  const auto total_query_start = std::chrono::system_clock::now();

  size_t query_count = 0;

  while (true) {
    auto query_batch_items = query_batch_provider->ReadNextBatch(nr);

    spu::psi::AllGatherItemsSize(link_ctx, query_batch_items.size());

    if (query_batch_items.empty()) {
      break;
    }

    const auto oprf_start = std::chrono::system_clock::now();
    std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>
        items_oprf = receiver.RequestOPRF(query_batch_items, link_ctx);

    const auto oprf_end = std::chrono::system_clock::now();
    const DurationMillis oprf_duration = oprf_end - oprf_start;
    SPDLOG_INFO("*** server oprf duration:{}", oprf_duration.count());

    const auto query_start = std::chrono::system_clock::now();
    std::pair<std::vector<size_t>, std::vector<std::string>> query_result =
        receiver.RequestQuery(items_oprf.first, items_oprf.second, link_ctx);

    const auto query_end = std::chrono::system_clock::now();
    const DurationMillis query_duration = query_end - query_start;
    SPDLOG_INFO("*** server query duration:{}", query_duration.count());

    SPDLOG_INFO("query_result size:{}", query_result.first.size());

    yacl::io::ColumnVectorBatch batch;

    std::vector<std::vector<std::string>> query_id_results(ids.size());
    std::vector<std::vector<std::string>> query_label_results(
        label_columns_name.size());

    for (size_t i = 0; i < query_result.first.size(); ++i) {
      std::vector<std::string> result_ids =
          absl::StrSplit(query_batch_items[query_result.first[i]], ",");

      SPU_ENFORCE(result_ids.size() == ids.size());

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

    for (size_t i = 0; i < ids.size(); ++i) {
      batch.AppendCol(query_id_results[i]);
    }
    for (size_t i = 0; i < label_columns_name.size(); ++i) {
      batch.AppendCol(query_label_results[i]);
    }

    writer.Add(batch);

    query_count++;
  }

  writer.Close();

  SPDLOG_INFO("query_count:{}", query_count);

  const auto total_query_end = std::chrono::system_clock::now();
  const DurationMillis total_query_duration =
      total_query_end - total_query_start;
  SPDLOG_INFO("*** total query duration:{}", total_query_duration.count());

  return 0;
}
