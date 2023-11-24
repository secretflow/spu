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

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"

#include "libspu/core/value.h"
#include "libspu/device/api.h"
#include "libspu/device/debug_dump_constant.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/device/symbol_table.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/simulate.h"

llvm::cl::opt<std::string> SnapshotDir(
    "snapshot_dir", llvm::cl::desc("folder contains core snapshot files"),
    llvm::cl::init("."));

// Mode switch
llvm::cl::opt<bool> LocalMode("local", llvm::cl::desc("local simulation mode"),
                              llvm::cl::init(false));

// Network only settings
llvm::cl::opt<std::string> Parties(
    "parties",
    llvm::cl::init("127.0.0.1:61530,127.0.0.1:61531,127.0.0.1:61532"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

// Local simulation only settings
llvm::cl::opt<uint32_t> NumProc(
    "num_processor",
    llvm::cl::desc("number of processors to create (local simulation only)"),
    llvm::cl::init(3));

std::shared_ptr<yacl::link::Context> MakeLink(const std::string &parties,
                                              size_t rank) {
  yacl::link::ContextDesc lctx_desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (auto &host : hosts) {
    const auto id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, host});
  }
  auto lctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

std::unique_ptr<spu::SPUContext> MakeSPUContext(
    const spu::RuntimeConfig &config) {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());

  return std::make_unique<spu::SPUContext>(config, lctx);
}

spu::RuntimeConfig parseRuntimeConfig(
    const std::filesystem::path &snapshot_dir) {
  auto config_file = spu::device::getConfigFilePath(snapshot_dir);
  SPU_ENFORCE(std::filesystem::exists(config_file),
              "Serialized config file {} does not exit", config_file.c_str());
  SPDLOG_INFO("Read config file from {}", config_file.c_str());
  std::ifstream stream(config_file, std::ios::binary);

  spu::RuntimeConfig config;
  SPU_ENFORCE(config.ParseFromIstream(&stream),
              "Parse serialized config file {} failed", config_file.c_str());
  return config;
}

spu::ExecutableProto parseExecutable(
    const std::filesystem::path &snapshot_dir) {
  auto code_file = spu::device::getCodeFilePath(snapshot_dir);
  SPU_ENFORCE(std::filesystem::exists(code_file),
              "Serialized executable file {} does not exit", code_file.c_str());
  SPDLOG_INFO("Read config file from {}", code_file.c_str());
  std::ifstream stream(code_file, std::ios::binary);

  spu::ExecutableProto code;
  SPU_ENFORCE(code.ParseFromIstream(&stream),
              "Parse serialized code file {} failed", code_file.c_str());
  return code;
}

spu::device::SymbolTable parseSymbolTable(
    const std::filesystem::path &snapshot_dir) {
  auto data_dir = spu::device::getRankFolder(snapshot_dir, Rank.getValue());
  SPU_ENFORCE(std::filesystem::exists(data_dir),
              "Serialized data dir {} does not exit", data_dir.c_str());
  SPDLOG_INFO("Read inputs file from {}", data_dir.c_str());

  spu::device::SymbolTable table;

  for (const auto &file : std::filesystem::directory_iterator(data_dir)) {
    const auto &filename = file.path().filename();

    if (filename.extension() == spu::device::getMetaExtension()) {
      spu::ValueProto vp;
      {
        SPDLOG_INFO("Read inputs meta {}", file.path().c_str());
        std::ifstream stream(file.path(), std::ios::binary);
        vp.meta.ParseFromIstream(&stream);
      }
      const auto var_name = filename.stem().native();
      // Get slices
      int64_t counter = 0;
      while (true) {
        auto chunk_file = spu::device::getValueChunkFilePath(
            snapshot_dir, Rank.getValue(), var_name, counter);
        if (std::filesystem::exists(chunk_file)) {
          SPDLOG_INFO("Read inputs data chunk {}", chunk_file.c_str());
          std::ifstream stream(chunk_file, std::ios::binary);
          vp.chunks.resize(counter + 1);
          vp.chunks[counter].ParseFromIstream(&stream);
          ++counter;
        } else {
          break;
        }
      }

      table.setVar(var_name, spu::Value::fromProto(vp));
    }
  }

  return table;
}

void RpcBasedRunner(const std::filesystem::path &snapshot_dir) {
  auto sctx = MakeSPUContext(parseRuntimeConfig(snapshot_dir));

  spu::device::SymbolTable table = parseSymbolTable(snapshot_dir);

  spu::device::pphlo::PPHloExecutor executor;

  SPDLOG_INFO("Run with config {}", sctx->config().DebugString());

  spu::device::execute(&executor, sctx.get(), parseExecutable(snapshot_dir),
                       &table);
}

void MemBasedRunner(const std::filesystem::path &snapshot_dir) {
  auto world_size = NumProc.getValue();

  SPDLOG_INFO("world size = {}", world_size);

  auto rt_config = parseRuntimeConfig(snapshot_dir);
  rt_config.set_enable_runtime_snapshot(false);

  spu::mpc::utils::simulate(
      world_size, [&](const std::shared_ptr<::yacl::link::Context> &lctx) {
        spu::SPUContext sctx(rt_config, lctx);

        spu::mpc::Factory::RegisterProtocol(&sctx, sctx.lctx());

        spu::device::pphlo::PPHloExecutor executor;

        auto executable = parseExecutable(snapshot_dir);
        spu::device::SymbolTable table = parseSymbolTable(snapshot_dir);

        spu::device::execute(&executor, &sctx, executable, &table);
      });
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::filesystem::path snapshot_dir = SnapshotDir.getValue();

  auto local = LocalMode.getValue();

  if (local) {
    MemBasedRunner(snapshot_dir);
  } else {
    RpcBasedRunner(snapshot_dir);
  }
}
