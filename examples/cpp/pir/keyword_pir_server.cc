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
// build keyword_pir_server
// > bazel build //examples/cpp/pir:keyword_pir_server -c opt
//
// To run the example, start terminals:
// > ./keyword_pir_server -rank 0 -setup_path pir_setup_dir 
//         -oprfkey_path secret_key.bin -data_per_query 256 -label_max_len 40  
//         -params_path psi_params.bin -label_columns label
// clang-format on

#include <chrono>
#include <filesystem>
#include <string>

#include "examples/cpp/pir/keyword_pir_utils.h"
#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/core/labeled_psi/sender.h"
#include "libspu/psi/utils/cipher_store.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> InPathOpt(
    "in_path", llvm::cl::init("data.csv"),
    llvm::cl::desc("[in] pir data in file path"));

llvm::cl::opt<std::string> OprfKeyPathOpt(
    "oprfkey_path", llvm::cl::init("oprf_key.bin"),
    llvm::cl::desc("[in] ecc oprf secretkey file path, 32bytes binary file"));

llvm::cl::opt<int> DataPerQueryOpt("data_per_query", llvm::cl::init(256),
                                   llvm::cl::desc("data count per query"));

llvm::cl::opt<int> LabelPadLengthOpt(
    "label_max_len", llvm::cl::init(288),
    llvm::cl::desc("pad label data to max len"));

llvm::cl::opt<std::string> LabelsColumnsOpt("label_columns",
                                            llvm::cl::init("label"),
                                            llvm::cl::desc("label columns"));

llvm::cl::opt<std::string> SetupPathOpt(
    "setup_path", llvm::cl::init("."),
    llvm::cl::desc("[in] db setup data path"));

llvm::cl::opt<std::string> ParamsPathOpt(
    "params_path", llvm::cl::init("params.bin"),
    llvm::cl::desc("[in] params output file path"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  SPDLOG_INFO("setup");

  auto hctx = MakeHalContext();
  auto link_ctx = hctx->lctx();

  apsi::PSIParams psi_params =
      spu::pir::examples::utils::ReadPsiParams(ParamsPathOpt.getValue());

  std::vector<uint8_t> oprf_key =
      spu::pir::examples::utils::ReadEcSecretKeyFile(OprfKeyPathOpt.getValue());

  SPDLOG_INFO("table_params hash_func_count:{}",
              psi_params.table_params().hash_func_count);

  size_t label_byte_count = LabelPadLengthOpt.getValue();
  size_t nonce_byte_count = 16;

  bool compressed = false;
  std::shared_ptr<spu::psi::SenderDB> sender_db =
      std::make_shared<spu::psi::SenderDB>(
          psi_params, oprf_key, SetupPathOpt.getValue(), label_byte_count,
          nonce_byte_count, compressed);

  SPDLOG_INFO("db GetItemCount:{}", sender_db->GetItemCount());

  SPDLOG_INFO("db, bin_bundle_count:{}, packing_rate:{}",
              sender_db->GetBinBundleCount(), sender_db->GetPackingRate());

  std::vector<std::string> labels =
      absl::StrSplit(LabelsColumnsOpt.getValue(), ',');
  SPU_ENFORCE(labels.size() > 0);

  yacl::Buffer labels_buffer = spu::psi::utils::SerializeStrItems(labels);
  // send labels column name
  link_ctx->SendAsync(link_ctx->NextRank(), labels_buffer,
                      fmt::format("send label columns name"));

  // send psi params
  yacl::Buffer params_buffer = spu::psi::PsiParamsToBuffer(psi_params);
  link_ctx->SendAsync(link_ctx->NextRank(), params_buffer,
                      fmt::format("send psi params"));

  const auto total_query_start = std::chrono::system_clock::now();

  size_t query_count = 0;
  while (true) {
    // oprf
    const auto oprf_start = std::chrono::system_clock::now();

    std::vector<size_t> batch_data_size =
        spu::psi::AllGatherItemsSize(link_ctx, 0);

    SPDLOG_INFO("client data size: {}", batch_data_size[link_ctx->NextRank()]);
    if (batch_data_size[link_ctx->NextRank()] == 0) {
      break;
    }

    std::unique_ptr<spu::psi::IEcdhOprfServer> oprf_server =
        spu::psi::CreateEcdhOprfServer(oprf_key, spu::psi::OprfType::Basic,
                                       spu::psi::CurveType::CURVE_FOURQ);

    spu::psi::LabelPsiSender sender(sender_db);

    sender.RunOPRF(std::move(oprf_server), link_ctx);

    const auto oprf_end = std::chrono::system_clock::now();
    const DurationMillis oprf_duration = oprf_end - oprf_start;
    SPDLOG_INFO("*** server oprf duration:{}", oprf_duration.count());

    const auto query_start = std::chrono::system_clock::now();

    sender.RunQuery(link_ctx);

    const auto query_end = std::chrono::system_clock::now();
    const DurationMillis query_duration = query_end - query_start;
    SPDLOG_INFO("*** server query duration:{}", query_duration.count());

    query_count++;
  }
  SPDLOG_INFO("query_count:{}", query_count);

  const auto total_query_end = std::chrono::system_clock::now();
  const DurationMillis total_query_duration =
      total_query_end - total_query_start;
  SPDLOG_INFO("*** total query duration:{}", total_query_duration.count());

  return 0;
}