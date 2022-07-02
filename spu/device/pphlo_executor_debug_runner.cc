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

#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"
#include "yasl/link/test_util.h"

#include "spu/device/pphlo_executor.h"
#include "spu/device/symbol_table.h"
#include "spu/device/test_utils.h"
#include "spu/hal/debug.h"
#include "spu/hal/value.h"
#include "spu/mpc/util/simulate.h"

llvm::cl::opt<std::string>
    DumpDir("dump_dir", llvm::cl::desc("folder contains core dump files"),
            llvm::cl::init("."));
llvm::cl::opt<uint32_t>
    NumProc("num_processor", llvm::cl::desc("number of processors to create"),
            llvm::cl::init(2));
llvm::cl::opt<std::string> Field("field", llvm::cl::desc("ring field size"),
                                 llvm::cl::init("FM128"));
llvm::cl::opt<std::string> Protocol("protocol", llvm::cl::desc("protocol kind"),
                                    llvm::cl::init("SEMI2K"));

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

  size_t world_size = NumProc.getValue();
  std::vector<spu::device::SymbolTable> tables(world_size);

  for (size_t idx = 0; idx < world_size; ++idx) {
    for (int var_counter = 0; var_counter < exec.input_names_size();
         ++var_counter) {
      auto data_file =
          core_dump_dir / fmt::format("processor{}{}.txt", idx, var_counter);

      YASL_ENFORCE(std::filesystem::exists(data_file),
                   "Data file does not exist");

      std::ifstream stream(data_file, std::ios::binary);

      SPDLOG_INFO("Read variable {} for processor {} from {}",
                  exec.input_names(var_counter), idx, data_file.c_str());

      spu::ValueProto vp;
      YASL_ENFORCE(vp.ParseFromIstream(&stream));
      tables[idx].setVar(exec.input_names(var_counter),
                         spu::hal::Value::fromProto(vp));
    }
  }

  spu::RuntimeConfig config;

  // Parse field
  spu::FieldType field;
  YASL_ENFORCE(spu::FieldType_Parse(Field.getValue(), &field),
               "Invalid field {}", Field.getValue());
  config.set_field(field);

  // Parse protocol
  spu::ProtocolKind protocol;
  YASL_ENFORCE(spu::ProtocolKind_Parse(Protocol.getValue(), &protocol),
               "Invalid protocol kind {}", Protocol.getValue());
  config.set_protocol(protocol);

  config.set_enable_pphlo_trace(true);

  ::spu::mpc::util::simulate(
      world_size, [&](const std::shared_ptr<::yasl::link::Context> &lctx) {
        ::spu::HalContext hctx(config, lctx);
        ::spu::device::PPHloExecutor executor(&hctx);
        auto *env = &tables[lctx->Rank()];
        executor.runWithEnv(exec, env);
      });
}
