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
#include <iostream>

#include "llvm/Support/CommandLine.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/compile.h"

llvm::cl::opt<std::string>
    InputFilename("in", llvm::cl::desc("Specify input filename"),
                  llvm::cl::value_desc("filename"));

llvm::cl::opt<std::string>
    PrettyPrintDir("pppath", llvm::cl::desc("Location to dump pretty print"),
                   llvm::cl::value_desc("path"));

llvm::cl::opt<std::string>
    OutputFilename("out", llvm::cl::desc("Specify output filename"),
                   llvm::cl::value_desc("filename"));

llvm::cl::opt<std::string>
    InputVisibility("invis", llvm::cl::desc("Inputs visibility"),
                    llvm::cl::value_desc("input visibility"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Check file existence
  std::filesystem::path in_xla(InputFilename.getValue());

  auto context = std::make_unique<spu::compiler::CompilationContext>();

  if (!PrettyPrintDir.getValue().empty()) {
    context->enablePrettyPrintWithDir(PrettyPrintDir.getValue());
  }
  if (!InputVisibility.getValue().empty()) {
    context->setInputVisibilityString(InputVisibility.getValue());
  }

  auto ret = spu::compiler::compile(context.get(), in_xla, "hlo");

  if (OutputFilename.empty()) {
    std::cout << ret << std::endl;
  } else {
    std::ofstream out(OutputFilename.getValue());
    out << ret;
  }

  return 0;
}
