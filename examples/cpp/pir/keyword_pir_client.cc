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
// >       -key_columns id -out_path pir_out.csv
// clang-format on

#include <chrono>
#include <filesystem>
#include <string>

#include "examples/cpp/utils.h"
#include "yacl/io/rw/csv_writer.h"

#include "libspu/pir/pir.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

#include "libspu/pir/pir.pb.h"

using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> InPathOpt("in_path", llvm::cl::init("data.csv"),
                                     llvm::cl::desc("pir data in file path"));

llvm::cl::opt<std::string> KeyColumnsOpt("key_columns", llvm::cl::init("id"),
                                         llvm::cl::desc("key columns"));

llvm::cl::opt<std::string> OutPathOpt(
    "out_path", llvm::cl::init("."),
    llvm::cl::desc("[out] pir query output path for db setup data"));

namespace {

constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;

}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto sctx = MakeSPUContext();
  auto link_ctx = sctx->lctx();

  link_ctx->SetRecvTimeout(kLinkRecvTimeout);

  std::vector<std::string> ids = absl::StrSplit(KeyColumnsOpt.getValue(), ',');

  spu::pir::PirClientConfig config;

  config.set_pir_protocol(spu::pir::PirProtocol::KEYWORD_PIR_LABELED_PSI);

  config.set_input_path(InPathOpt.getValue());
  config.mutable_key_columns()->Add(ids.begin(), ids.end());
  config.set_output_path(OutPathOpt.getValue());

  spu::pir::PirResultReport report = spu::pir::PirClient(link_ctx, config);

  SPDLOG_INFO("data count:{}", report.data_count());

  return 0;
}
