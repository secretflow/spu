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
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/device/symbol_table.h"
#include "libspu/device/test_utils.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/mpc/utils/simulate.h"

llvm::cl::opt<std::string> SnapshotDir(
    "snapshot_dir", llvm::cl::desc("folder contains core snapshot files"),
    llvm::cl::init("."));

// Mode switch
llvm::cl::opt<bool> LocalMode("local", llvm::cl::desc("local simulation mode"),
                              llvm::cl::init(false));

// Network only settings
llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531,127.0.0.1:9532"),
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
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const auto id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }
  auto lctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

std::unique_ptr<spu::SPUContext> MakeSPUContext(
    const spu::device::SnapshotProto &snapshot) {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());

  return std::make_unique<spu::SPUContext>(snapshot.runtime_cfg(), lctx);
}

spu::device::SnapshotProto ParseSnapshotFile(
    const std::filesystem::path &snapshot_file) {
  spu::device::SnapshotProto snapshot;
  {
    SPU_ENFORCE(std::filesystem::exists(snapshot_file),
                "Serialized snapshot file {} does not exit",
                snapshot_file.c_str());
    SPDLOG_INFO("Read snapshot file from {}", snapshot_file.c_str());
    std::ifstream stream(snapshot_file, std::ios::binary);
    SPU_ENFORCE(snapshot.ParseFromIstream(&stream),
                "Parse serialized snapshot file {} failed",
                snapshot_file.c_str());
  }

  return snapshot;
}

void RpcBasedRunner(const std::filesystem::path &snapshot_dir) {
  auto snapshot_file =
      snapshot_dir / fmt::format("snapshot_{}.spu", Rank.getValue());
  spu::device::SnapshotProto snapshot = ParseSnapshotFile(snapshot_file);
  auto sctx = MakeSPUContext(snapshot);

  spu::device::SymbolTable table =
      spu::device::SymbolTable::fromProto(snapshot.environ());

  spu::device::pphlo::PPHloExecutor executor;

  SPDLOG_INFO("Run with config {}", sctx->config().DebugString());

  spu::device::execute(&executor, sctx.get(), snapshot.executable(), &table);
}

void MemBasedRunner(const std::filesystem::path &snapshot_dir) {
  auto world_size = NumProc.getValue();

  spu::mpc::utils::simulate(
      world_size, [&](const std::shared_ptr<::yacl::link::Context> &lctx) {
        auto snapshot_file =
            snapshot_dir / fmt::format("snapshot_{}.spu", lctx->Rank());

        spu::device::SnapshotProto snapshot = ParseSnapshotFile(snapshot_file);

        spu::SPUContext sctx(snapshot.runtime_cfg(), lctx);

        spu::device::pphlo::PPHloExecutor executor;
        spu::device::SymbolTable table =
            spu::device::SymbolTable::fromProto(snapshot.environ());
        spu::device::execute(&executor, &sctx, snapshot.executable(), &table);
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
