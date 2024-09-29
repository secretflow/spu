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

#include "libspu/compiler/codegen/codegen.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/core/prelude.h"
#include "libspu/dialect/ring/IR/version.h"
#include "libspu/version.h"

namespace spu::compiler {

std::string CodeGen::doit(mlir::ModuleOp module) {
  // Add ir version attr
  module->setAttr("ring.version",
                  mlir::StringAttr::get(module->getContext(), getVersionStr()));
  // Emit module
  std::string ir_dump;
  llvm::raw_string_ostream stream(ir_dump);

  if (ctx_->emitByteCode()) {
    auto producer = fmt::format("spu_v{}", getVersionStr());
    mlir::BytecodeWriterConfig writerConfig(producer);

    writerConfig.setDesiredBytecodeVersion(
        mlir::spu::ring::RingDialectVersion::getBytecodeVersion());

    if (mlir::failed(writeBytecodeToFile(module, stream, writerConfig))) {
      SPU_THROW("failed to dump module to bytecode");
    }
  } else {
    module.print(stream);
  }

  return stream.str();
}

} // namespace spu::compiler
