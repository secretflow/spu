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
#include "yasl/link/test_util.h"

#include "spu/device/pphlo/executor.h"
#include "spu/device/symbol_table.h"
#include "spu/device/test_utils.h"
#include "spu/hal/debug.h"
#include "spu/hal/value.h"
#include "spu/mpc/util/simulate.h"

llvm::cl::opt<std::string>
    DumpDir("dump_dir", llvm::cl::desc("folder contains core dump files"),
            llvm::cl::init("."));

llvm::cl::opt<std::string> Field("field", llvm::cl::desc("ring field size"),
                                 llvm::cl::init("FM64"));
llvm::cl::opt<std::string> Protocol("protocol", llvm::cl::desc("protocol kind"),
                                    llvm::cl::init("ABY3"));

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531,127.0.0.1:9532"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

std::shared_ptr<yasl::link::Context> MakeLink(const std::string &parties,
                                              size_t rank) {
  yasl::link::ContextDesc lctx_desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const auto id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }
  auto lctx = yasl::link::FactoryBrpc().CreateContext(lctx_desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

std::unique_ptr<spu::HalContext> MakeHalContext() {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());

  spu::RuntimeConfig config;

  // Parse protocol
  spu::ProtocolKind pk;
  YASL_ENFORCE(spu::ProtocolKind_Parse(Protocol.getValue(), &pk),
               "Invalid protocol kind {}", Protocol.getValue());

  // Parse field
  spu::FieldType field;
  YASL_ENFORCE(spu::FieldType_Parse(Field.getValue(), &field),
               "Invalid field {}", Field.getValue());

  config.set_protocol(pk);
  config.set_field(field);
  config.set_enable_pphlo_trace(true);
  config.set_enable_type_checker(true);

  return std::make_unique<spu::HalContext>(config, lctx);
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::filesystem::path core_dump_dir = DumpDir.getValue();

  spu::ExecutableProto exec;
  {
    auto exec_file = core_dump_dir / "exec.txt";
    YASL_ENFORCE(std::filesystem::exists(exec_file),
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

  spu::device::SymbolTable table;
  auto hctx = MakeHalContext();
  spu::device::pphlo::PPHloExecutor executor(hctx.get());

  SPDLOG_INFO("Run with config {}", hctx->rt_config().DebugString());

  for (int var_counter = 0; var_counter < exec.input_names_size();
       ++var_counter) {
    auto data_file =
        core_dump_dir /
        fmt::format("data_{}_{}.txt", hctx->lctx()->Rank(), var_counter);

    YASL_ENFORCE(std::filesystem::exists(data_file),
                 "Data file does not exist");

    std::ifstream stream(data_file, std::ios::binary);

    spu::ValueProto vp;
    YASL_ENFORCE(vp.ParseFromIstream(&stream));
    auto v = spu::hal::Value::fromProto(vp);
    SPDLOG_INFO("Read input {} {} for processor {} from {}, v = {}",
                var_counter, exec.input_names(var_counter),
                hctx->lctx()->Rank(), data_file.c_str(), v);
    table.setVar(exec.input_names(var_counter), v);
  }
  executor.runWithEnv(exec, &table);
}
