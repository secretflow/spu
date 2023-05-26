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
// > bazel run //examples/cpp:simple_psi -- -rank 0 -protocol 1 -in_path examples/data/psi_1.csv -field_names id -out_path /tmp/p1.out
// > bazel run //examples/cpp:simple_psi -- -rank 1 -protocol 1 -in_path examples/data/psi_2.csv -field_names id -out_path /tmp/p2.out
// clang-format on

#include "absl/strings/str_split.h"
#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"

#include "libspu/psi/bucket_psi.h"

llvm::cl::opt<int> ProtocolOpt(
    "protocol", llvm::cl::init(1),
    llvm::cl::desc("select psi protocol, see `spu/psi/psi.proto`"));

llvm::cl::opt<std::string> InPathOpt("in_path", llvm::cl::init("data.csv"),
                                     llvm::cl::desc("psi data in file path "));

llvm::cl::opt<std::string> FieldNamesOpt("field_names", llvm::cl::init("id"),
                                         llvm::cl::desc("field names "));

llvm::cl::opt<std::string> OutPathOpt("out_path", llvm::cl::init(""),
                                      llvm::cl::desc("psi out file path"));

llvm::cl::opt<bool> ShouldSortOpt("should_sort", llvm::cl::init(false),
                                  llvm::cl::desc("whether sort psi result"));

llvm::cl::opt<bool> PrecheckOpt(
    "precheck_input", llvm::cl::init(false),
    llvm::cl::desc("whether precheck input dataset"));

llvm::cl::opt<int> BucketSizeOpt("bucket_size", llvm::cl::init(1 << 20),
                                 llvm::cl::desc("hash bucket size"));

llvm::cl::opt<double> DPPsiBobSubSamplingOpt(
    "bob_sub_sampling", llvm::cl::init(0.9),
    llvm::cl::desc("dppsi bob_sub_sampling"));

llvm::cl::opt<double> DPPsiEpsilonOpt("epsilon", llvm::cl::init(3),
                                      llvm::cl::desc("dppsi epsilon"));

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto sctx = MakeSPUContext();

  auto field_list = absl::StrSplit(FieldNamesOpt.getValue(), ',');

  spu::psi::BucketPsiConfig config;
  config.mutable_input_params()->set_path(InPathOpt.getValue());
  config.mutable_input_params()->mutable_select_fields()->Add(
      field_list.begin(), field_list.end());
  config.mutable_input_params()->set_precheck(PrecheckOpt.getValue());
  config.mutable_output_params()->set_path(OutPathOpt.getValue());
  config.mutable_output_params()->set_need_sort(ShouldSortOpt.getValue());
  config.set_psi_type(static_cast<spu::psi::PsiType>(ProtocolOpt.getValue()));
  config.set_receiver_rank(0);

  if (spu::psi::DP_PSI_2PC == ProtocolOpt.getValue()) {
    spu::psi::DpPsiParams* dppsi_params = config.mutable_dppsi_params();
    dppsi_params->set_bob_sub_sampling(DPPsiBobSubSamplingOpt.getValue());
    dppsi_params->set_epsilon(DPPsiEpsilonOpt.getValue());
  }

  // one-way PSI, just one party get result
  config.set_broadcast_result(false);
  config.set_bucket_size(BucketSizeOpt.getValue());
  config.set_curve_type(spu::psi::CurveType::CURVE_25519);

  try {
    spu::psi::BucketPsi bucket_psi(config, sctx->lctx());
    auto report = bucket_psi.Run();

    SPDLOG_INFO("rank:{} original_count:{} intersection_count:{}",
                sctx->lctx()->Rank(), report.original_count(),
                report.intersection_count());
  } catch (const std::exception& e) {
    SPDLOG_ERROR("run psi failed: {}", e.what());
    return -1;
  }

  return 0;
}
