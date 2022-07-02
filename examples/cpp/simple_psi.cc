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

// clang-format off
// To run the example, start two terminals:
// > bazel run //examples/cpp:simple_psi -- -rank 0 -protocol ecdh -in_path examples/data/psi_1.csv -field_names id -out_path /tmp/p1.out 
// > bazel run //examples/cpp:simple_psi -- -rank 1 -protocol ecdh -in_path examples/data/psi_2.csv -field_names id -out_path /tmp/p2.out
// clang-format on

#include "absl/strings/str_split.h"
#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"

#include "spu/psi/psi.h"

llvm::cl::opt<std::string> ProtocolOpt(
    "protocol", llvm::cl::init("ecdh"),
    llvm::cl::desc("select psi protocol ecdh/kkrt"));

llvm::cl::opt<std::string> InPathOpt("in_path", llvm::cl::init("data.csv"),
                                     llvm::cl::desc("psi data in file path "));

llvm::cl::opt<std::string> FieldNamesOpt("field_names", llvm::cl::init("id"),
                                         llvm::cl::desc("field names "));

llvm::cl::opt<std::string> OutPathOpt("out_path", llvm::cl::init(""),
                                      llvm::cl::desc("psi out file path"));

llvm::cl::opt<bool> ShouldSortOpt("should_sort", llvm::cl::init(false),
                                  llvm::cl::desc("whether sort psi result"));

llvm::cl::opt<uint32_t> NumBinsOpt("num_bins", llvm::cl::init(0),
                                   llvm::cl::desc("number of bins"));

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto hctx = MakeHalContext();

  spu::psi::LegacyPsiOptions psi_options;

  psi_options.base_options.link_ctx = hctx->lctx();
  psi_options.base_options.in_path = InPathOpt.getValue();
  psi_options.base_options.field_names =
      absl::StrSplit(FieldNamesOpt.getValue(), ',');
  psi_options.base_options.out_path = OutPathOpt.getValue();
  psi_options.base_options.should_sort = ShouldSortOpt.getValue();

  psi_options.psi_protocol = ProtocolOpt.getValue();
  psi_options.num_bins = NumBinsOpt.getValue();

  if (psi_options.psi_protocol == spu::psi::kPsiProtocolKkrt) {
    psi_options.broadcast_result = false;
  }

  std::shared_ptr<spu::psi::PsiExecutorBase> psi_executor =
      spu::psi::BuildPsiExecutor(psi_options);

  spu::psi::PsiReport psi_report;

  psi_executor->Init();

  psi_executor->Run(&psi_report);

  SPDLOG_INFO("original_count:{} intersection_count:{}",
              psi_report.original_count, psi_report.intersection_count);

  return 0;
}