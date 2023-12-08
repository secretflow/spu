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

#include "libspu/compiler/compile.h"

#include "mlir/IR/BuiltinOps.h"

#include "libspu/compiler/codegen/codegen.h"
#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/core/core.h"
#include "libspu/compiler/front_end/fe.h"

namespace spu::compiler {

std::string compile(const CompilationSource &source,
                    const CompilerOptions &copts) {
  CompilationContext ctx(copts);

  // Call front end
  FE fe(&ctx);
  auto mlir_module = fe.doit(source);

  // Run core passes
  Core core(&ctx);
  core.doit(mlir_module.get());

  // Run codegen
  return spu::compiler::CodeGen::doit(mlir_module.get());
}

} // namespace spu::compiler
