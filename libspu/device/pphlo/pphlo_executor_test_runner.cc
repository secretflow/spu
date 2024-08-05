// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/device/pphlo/pphlo_executor_test_runner.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/compile.h"
#include "libspu/device/api.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::device::pphlo::test {

Runner::Runner(size_t world_size, FieldType field, ProtocolKind protocol)
    : world_size_(world_size) {
  config_.set_field(field);
  config_.set_protocol(protocol);
  config_.set_enable_type_checker(true);
  io_ = std::make_unique<LocalIo>(world_size_, config_);
}

std::string Runner::compileMHlo(const std::string &mhlo,
                                const std::vector<spu::Visibility> &vis) {
  CompilationSource source;
  source.set_ir_type(SourceIRType::MLIR_HLO);
  source.set_ir_txt(mhlo);
  for (const auto v : vis) {
    source.add_input_visibility(v);
  }

  compiler::CompilationContext ctx;
  return compiler::compile(&ctx, source.SerializeAsString());
}

void Runner::run(const std::string &mlir, size_t num_output) {
  for (size_t idx = 0; idx < num_output; ++idx) {
    executable_.add_output_names(fmt::format("output{}", idx));
  }
  executable_.set_code(mlir);
  ::spu::mpc::utils::simulate(
      world_size_, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig conf;
        conf.CopyFrom(config_);
        if (lctx->Rank() == 0) {
          // conf.set_enable_action_trace(true);
        }
        SPUContext sctx = kernel::test::makeSPUContext(conf, lctx);
        auto *env = io_->GetSymbolTable(lctx->Rank());
        pphlo::PPHloExecutor executor;
        execute(&executor, &sctx, executable_, env);
      });
}

}  // namespace spu::device::pphlo::test
