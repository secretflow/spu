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
#include "yacl/link/test_util.h"

#include "spu/device/api.h"
#include "spu/device/pphlo/pphlo_executor.h"
#include "spu/device/symbol_table.h"
#include "spu/device/test_utils.h"
#include "spu/kernel/hal/debug.h"
#include "spu/kernel/value.h"
#include "spu/mpc/util/simulate.h"

llvm::cl::opt<std::string>
    DumpDir("dump_dir", llvm::cl::desc("folder contains core dump files"),
            llvm::cl::init("."));

llvm::cl::opt<std::string> Field("field", llvm::cl::desc("ring field size"),
                                 llvm::cl::init("FM64"));
llvm::cl::opt<std::string> Protocol("protocol", llvm::cl::desc("protocol kind"),
                                    llvm::cl::init("ABY3"));

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

spu::RuntimeConfig CreateRuntimeConfig() {
  spu::RuntimeConfig config;

  // Parse protocol
  spu::ProtocolKind pk;
  YACL_ENFORCE(spu::ProtocolKind_Parse(Protocol.getValue(), &pk),
               "Invalid protocol kind {}", Protocol.getValue());

  // Parse field
  spu::FieldType field;
  YACL_ENFORCE(spu::FieldType_Parse(Field.getValue(), &field),
               "Invalid field {}", Field.getValue());

  config.set_protocol(pk);
  config.set_field(field);
  config.set_enable_pphlo_trace(false);
  config.set_enable_pphlo_profile(true);

  return config;
}

std::unique_ptr<spu::HalContext> MakeHalContext() {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());

  spu::RuntimeConfig config = CreateRuntimeConfig();

  return std::make_unique<spu::HalContext>(config, lctx);
}

void RpcBasedRunner(spu::ExecutableProto &exec,
                    const std::filesystem::path &core_dump_dir) {
  spu::device::SymbolTable table;
  auto hctx = MakeHalContext();
  spu::device::pphlo::PPHloExecutor executor;

  SPDLOG_INFO("Run with config {}", hctx->rt_config().DebugString());

  for (int var_counter = 0; var_counter < exec.input_names_size();
       ++var_counter) {
    auto data_file =
        core_dump_dir /
        fmt::format("data_{}_{}.txt", hctx->lctx()->Rank(), var_counter);

    YACL_ENFORCE(std::filesystem::exists(data_file),
                 "Data file does not exist");

    std::ifstream stream(data_file, std::ios::binary);

    spu::ValueProto vp;
    YACL_ENFORCE(vp.ParseFromIstream(&stream));
    auto v = spu::Value::fromProto(vp);
    SPDLOG_INFO("Read input {} {} for processor {} from {}, v = {}",
                var_counter, exec.input_names(var_counter),
                hctx->lctx()->Rank(), data_file.c_str(), v);
    table.setVar(exec.input_names(var_counter), v);
  }
  spu::device::execute(&executor, hctx.get(), exec, &table);
}

void MemBasedRunner(spu::ExecutableProto &exec,
                    const std::filesystem::path &core_dump_dir) {
  auto world_size = NumProc.getValue();
  std::vector<spu::device::SymbolTable> tables(world_size);
  auto config = CreateRuntimeConfig();

  ::spu::mpc::util::simulate(
      world_size, [&](const std::shared_ptr<::yacl::link::Context> &lctx) {
        ::spu::HalContext hctx(config, lctx);
        ::spu::device::pphlo::PPHloExecutor executor;
        for (int var_counter = 0; var_counter < exec.input_names_size();
             ++var_counter) {
          auto data_file =
              core_dump_dir /
              fmt::format("data_{}_{}.txt", lctx->Rank(), var_counter);
          YACL_ENFORCE(std::filesystem::exists(data_file),
                       "Data file does not exist");
          std::ifstream stream(data_file, std::ios::binary);
          spu::ValueProto vp;
          YACL_ENFORCE(vp.ParseFromIstream(&stream));
          auto v = spu::Value::fromProto(vp);
          SPDLOG_INFO("Read input {} {} for processor {} from {}, v = {}",
                      var_counter, exec.input_names(var_counter), lctx->Rank(),
                      data_file.c_str(), v);
          tables[lctx->Rank()].setVar(exec.input_names(var_counter), v);
        }
        auto *env = &tables[lctx->Rank()];
        spu::device::execute(&executor, &hctx, exec, env);
      });
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::filesystem::path core_dump_dir = DumpDir.getValue();

  spu::ExecutableProto exec;
  {
    auto exec_file = core_dump_dir / "executable.txt";
    YACL_ENFORCE(std::filesystem::exists(exec_file),
                 "Serialized executable file does not exit");
    SPDLOG_INFO("Read executable file from {}", exec_file.c_str());
    std::ifstream stream(exec_file, std::ios::binary);
    if (!exec.ParseFromIstream(&stream)) {
      // Try raw mlir with 0 inputs
      // Rewind fp
      stream.clear();
      stream.seekg(0);
      exec.set_code(std::string((std::istreambuf_iterator<char>(stream)),
                                std::istreambuf_iterator<char>()));
    }
  }

  auto local = LocalMode.getValue();

  if (local) {
    MemBasedRunner(exec, core_dump_dir);
  } else {
    RpcBasedRunner(exec, core_dump_dir);
  }
}
