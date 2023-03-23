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
// build keyword_pir_setup
// > bazel build //examples/cpp/pir:keyword_pir_setup -c opt
//
// To generate ecc oprf secret key, start terminals:
// > dd if=/dev/urandom of=secret_key.bin bs=32 count=1
//
// To run the example, start terminals:
// > ./keyword_pir_setup -in_path ../../data/psi_server_data.csv -oprfkey_path secret_key.bin 
//      -key_columns id -label_columns label -data_per_query 256 -label_max_len 40 
//      -out_path pir_setup_dir -params_path psi_params.bin
// clang-format on

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "examples/cpp/pir/keyword_pir_utils.h"
#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/core/labeled_psi/sender.h"
#include "libspu/psi/utils/cipher_store.h"

using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> InPathOpt(
    "in_path", llvm::cl::init("data.csv"),
    llvm::cl::desc("[in] pir data in file path"));

llvm::cl::opt<std::string> OprfKeyPathOpt(
    "oprfkey_path", llvm::cl::init("oprf_key.bin"),
    llvm::cl::desc("[in] ecc oprf secretkey file path, 32bytes binary file"));

llvm::cl::opt<int> DataPerQueryOpt("data_per_query", llvm::cl::init(256),
                                   llvm::cl::desc("data count per query"));

llvm::cl::opt<std::string> KeyColumnsOpt("key_columns", llvm::cl::init("id"),
                                         llvm::cl::desc("key columns"));

llvm::cl::opt<std::string> LabelsColumnsOpt("label_columns",
                                            llvm::cl::init("label"),
                                            llvm::cl::desc("label columns"));

llvm::cl::opt<int> LabelPadLengthOpt(
    "label_max_len", llvm::cl::init(288),
    llvm::cl::desc("pad label data to max len"));

llvm::cl::opt<std::string> OutPathOpt(
    "out_path", llvm::cl::init("."),
    llvm::cl::desc("[out] output path for db setup data"));

llvm::cl::opt<std::string> ParamsOutPathOpt(
    "params_path", llvm::cl::init("params.bin"),
    llvm::cl::desc("[out] params output file path"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  SPDLOG_INFO("setup");

  std::vector<std::string> ids = absl::StrSplit(KeyColumnsOpt.getValue(), ',');
  std::vector<std::string> labels =
      absl::StrSplit(LabelsColumnsOpt.getValue(), ',');

  SPDLOG_INFO("in_path: {}", InPathOpt.getValue());
  SPDLOG_INFO("key columns: {}", KeyColumnsOpt.getValue());
  SPDLOG_INFO("label columns: {}", LabelsColumnsOpt.getValue());

  size_t nr = DataPerQueryOpt.getValue();
  size_t ns =
      spu::pir::examples::utils::CsvFileDataCount(InPathOpt.getValue(), ids);

  SPDLOG_INFO("nr:{}, ns:{}", nr, ns);

  apsi::PSIParams psi_params = spu::psi::GetPsiParams(nr, ns);

  spu::pir::examples::utils::WritePsiParams(ParamsOutPathOpt.getValue(),
                                            psi_params);

  std::vector<uint8_t> oprf_key =
      spu::pir::examples::utils::ReadEcSecretKeyFile(OprfKeyPathOpt.getValue());

  size_t label_byte_count = LabelPadLengthOpt.getValue();
  size_t nonce_byte_count = 16;

  std::string kv_store_path = OutPathOpt.getValue();
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

  bool compressed = false;
  std::shared_ptr<spu::psi::SenderDB> sender_db =
      std::make_shared<spu::psi::SenderDB>(psi_params, oprf_key, kv_store_path,
                                           label_byte_count, nonce_byte_count,
                                           compressed);

  const auto setdb_start = std::chrono::system_clock::now();

  std::shared_ptr<spu::psi::IBatchProvider> batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(InPathOpt.getValue(), ids,
                                                   labels);

  sender_db->SetData(batch_provider);

  SPDLOG_INFO("db GetItemCount:{}", sender_db->GetItemCount());
  SPDLOG_INFO("db, bin_bundle_count:{}, packing_rate:{}",
              sender_db->GetBinBundleCount(), sender_db->GetPackingRate());

  const auto setdb_end = std::chrono::system_clock::now();
  const DurationMillis setdb_duration = setdb_end - setdb_start;
  SPDLOG_INFO("*** step set db duration:{}", setdb_duration.count());

  return 0;
}
