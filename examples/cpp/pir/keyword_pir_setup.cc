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
// > ./keyword_pir_setup -in_path ../../data/psi_server_data.csv -oprf_key_path secret_key.bin
// >     -key_columns id -label_columns label -data_per_query 1 -label_max_len 40
// >     -setup_path pir_setup_dir
// clang-format on

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/pir/pir.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/core/labeled_psi/sender.h"

#include "libspu/pir/pir.pb.h"

using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> InPathOpt(
    "in_path", llvm::cl::init("data.csv"),
    llvm::cl::desc("[in] pir data in file path"));

llvm::cl::opt<std::string> OprfKeyPathOpt(
    "oprf_key_path", llvm::cl::init("oprf_key.bin"),
    llvm::cl::desc("[in] ecc oprf secretkey file path, 32bytes binary file"));

llvm::cl::opt<int> DataPerQueryOpt("count_per_query", llvm::cl::init(256),
                                   llvm::cl::desc("data count per query"));

llvm::cl::opt<std::string> KeyColumnsOpt("key_columns", llvm::cl::init("id"),
                                         llvm::cl::desc("key columns"));

llvm::cl::opt<std::string> LabelsColumnsOpt("label_columns",
                                            llvm::cl::init("label"),
                                            llvm::cl::desc("label columns"));

llvm::cl::opt<int> LabelPadLengthOpt(
    "max_label_length", llvm::cl::init(288),
    llvm::cl::desc("pad label data to max len"));

llvm::cl::opt<std::string> SetupPathOpt(
    "setup_path", llvm::cl::init("."),
    llvm::cl::desc("[out] output path for db setup data"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  SPDLOG_INFO("setup");

  std::vector<std::string> ids = absl::StrSplit(KeyColumnsOpt.getValue(), ',');
  std::vector<std::string> labels =
      absl::StrSplit(LabelsColumnsOpt.getValue(), ',');

  SPDLOG_INFO("in_path: {}", InPathOpt.getValue());
  SPDLOG_INFO("key columns: {}", KeyColumnsOpt.getValue());
  SPDLOG_INFO("label columns: {}", LabelsColumnsOpt.getValue());

  spu::pir::PirSetupConfig config;

  config.set_pir_protocol(spu::pir::PirProtocol::KEYWORD_PIR_LABELED_PSI);
  config.set_store_type(spu::pir::KvStoreType::LEVELDB_KV_STORE);
  config.set_input_path(InPathOpt.getValue());

  config.mutable_key_columns()->Add(ids.begin(), ids.end());
  config.mutable_label_columns()->Add(labels.begin(), labels.end());

  config.set_num_per_query(DataPerQueryOpt.getValue());
  config.set_label_max_len(LabelPadLengthOpt.getValue());
  config.set_oprf_key_path(OprfKeyPathOpt.getValue());
  config.set_setup_path(SetupPathOpt.getValue());

  spu::pir::PirResultReport report = spu::pir::PirSetup(config);

  SPDLOG_INFO("data count:{}", report.data_count());

  return 0;
}
