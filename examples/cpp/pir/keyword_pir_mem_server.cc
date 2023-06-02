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
// >        -oprf_key_path secret_key.bin
// clang-format on

#include <chrono>
#include <filesystem>
#include <string>

#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/pir/pir.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/core/labeled_psi/sender.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

#include "libspu/pir/pir.pb.h"

using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> InPathOpt(
    "in_path", llvm::cl::init("data.csv"),
    llvm::cl::desc("[in] pir data in file path"));

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

namespace {

constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;

}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  SPDLOG_INFO("setup");

  auto sctx = MakeSPUContext();
  auto link_ctx = sctx->lctx();

  link_ctx->SetRecvTimeout(kLinkRecvTimeout);

  std::vector<std::string> ids = absl::StrSplit(KeyColumnsOpt.getValue(), ',');
  std::vector<std::string> labels =
      absl::StrSplit(LabelsColumnsOpt.getValue(), ',');

  spu::pir::PirSetupConfig config;

  config.set_pir_protocol(spu::pir::PirProtocol::KEYWORD_PIR_LABELED_PSI);
  config.set_store_type(spu::pir::KvStoreType::LEVELDB_KV_STORE);
  config.set_input_path(InPathOpt.getValue());

  config.mutable_key_columns()->Add(ids.begin(), ids.end());
  config.mutable_label_columns()->Add(labels.begin(), labels.end());

  config.set_num_per_query(DataPerQueryOpt.getValue());
  config.set_label_max_len(LabelPadLengthOpt.getValue());
  config.set_oprf_key_path("");
  config.set_setup_path("::memory");

  spu::pir::PirResultReport report =
      spu::pir::PirMemoryServer(link_ctx, config);

  SPDLOG_INFO("data count:{}", report.data_count());

  return 0;
}
