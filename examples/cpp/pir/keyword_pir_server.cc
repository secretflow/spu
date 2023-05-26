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

llvm::cl::opt<std::string> OprfKeyPathOpt(
    "oprf_key_path", llvm::cl::init("oprf_key.bin"),
    llvm::cl::desc("[in] ecc oprf secretkey file path, 32bytes binary file"));

llvm::cl::opt<std::string> SetupPathOpt(
    "setup_path", llvm::cl::init("."),
    llvm::cl::desc("[in] db setup data path"));

namespace {

constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;

}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  SPDLOG_INFO("setup");

  auto sctx = MakeSPUContext();
  auto link_ctx = sctx->lctx();

  link_ctx->SetRecvTimeout(kLinkRecvTimeout);

  spu::pir::PirServerConfig config;

  config.set_pir_protocol(spu::pir::PirProtocol::KEYWORD_PIR_LABELED_PSI);
  config.set_store_type(spu::pir::KvStoreType::LEVELDB_KV_STORE);

  config.set_oprf_key_path(OprfKeyPathOpt.getValue());
  config.set_setup_path(SetupPathOpt.getValue());

  spu::pir::PirResultReport report = spu::pir::PirServer(link_ctx, config);

  SPDLOG_INFO("data count:{}", report.data_count());

  return 0;
}