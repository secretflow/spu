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
// build simple_dp_psi
// > bazel build //examples/cpp:simple_dp_psi -c opt
//
// To run the example, start two terminals:
// > ./simple_dp_psi -rank 0 -in_path examples/data/psi_1.csv -field_names id
// > ./simple_dp_psi -rank 1 -in_path examples/data/psi_2.csv -field_names id -out_path /tmp/p2.out
// clang-format on

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"

#include "libspu/psi/core/dp_psi/dp_psi.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/serialize.h"

namespace {

// dp_psi_params reference from paper
#if 0
std::map<size_t, DpPsiOptions> dp_psi_params_map = {
    {1 << 11, DpPsiOptions(0.9)},   {1 << 12, DpPsiOptions(0.9)},
    {1 << 13, DpPsiOptions(0.9)},   {1 << 14, DpPsiOptions(0.9)},
    {1 << 15, DpPsiOptions(0.9)},   {1 << 16, DpPsiOptions(0.9)},
    {1 << 17, DpPsiOptions(0.9)},   {1 << 18, DpPsiOptions(0.9)},
    {1 << 19, DpPsiOptions(0.9)},   {1 << 20, DpPsiOptions(0.995)},
    {1 << 21, DpPsiOptions(0.995)}, {1 << 22, DpPsiOptions(0.995)},
    {1 << 23, DpPsiOptions(0.995)}, {1 << 24, DpPsiOptions(0.995)},
    {1 << 25, DpPsiOptions(0.995)}, {1 << 26, DpPsiOptions(0.995)},
    {1 << 27, DpPsiOptions(0.995)}, {1 << 28, DpPsiOptions(0.995)},
    {1 << 29, DpPsiOptions(0.995)}, {1 << 30, DpPsiOptions(0.995)}};
#endif

spu::psi::DpPsiOptions GetDpPsiOptions(size_t items_size) {
  double p1 = 0.9;
  if (items_size > (1 << 19)) {
    p1 = 0.995;
  }

  return spu::psi::DpPsiOptions(p1);
}

void WriteCsvData(const std::string& file_name,
                  const std::vector<std::string>& items,
                  const std::vector<size_t>& intersection_idxes) {
  std::ofstream out_file;
  out_file.open(file_name, std::ios::app);

  for (auto intersection_idx : intersection_idxes) {
    out_file << items[intersection_idx] << std::endl;
  }

  out_file.close();
}

constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;
constexpr uint32_t kLinkWindowSize = 16;

}  // namespace

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

int main(int argc, char** argv) {  // NOLINT
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto sctx = MakeSPUContext();

  auto field_list = absl::StrSplit(FieldNamesOpt.getValue(), ',');
  auto batch_provider = std::make_shared<spu::psi::CsvBatchProvider>(
      InPathOpt.getValue(), field_list);

  std::vector<std::string> items;

  while (true) {
    auto batch_items = batch_provider->ReadNextBatch(4096);

    if (batch_items.empty()) {
      break;
    }
    items.insert(items.end(), batch_items.begin(), batch_items.end());
  }

  std::vector<size_t> intersection_idx;

  auto link_ctx = sctx->lctx();

  link_ctx->SetThrottleWindowSize(kLinkWindowSize);

  link_ctx->SetRecvTimeout(kLinkRecvTimeout);

  if (Rank.getValue() == 0) {
    yacl::Buffer bob_items_size_buffer = sctx->lctx()->Recv(
        link_ctx->NextRank(), fmt::format("peer items number"));
    size_t bob_items_size =
        spu::psi::utils::DeserializeSize(bob_items_size_buffer);

    spu::psi::DpPsiOptions options = GetDpPsiOptions(bob_items_size);

    size_t alice_sub_sample_size;
    size_t alice_up_sample_size;
    size_t intersection_size = spu::psi::RunDpEcdhPsiAlice(
        options, link_ctx, items, &alice_sub_sample_size,
        &alice_up_sample_size);

    SPDLOG_INFO("alice_sub_sample_size: {}", alice_sub_sample_size);
    SPDLOG_INFO("alice_up_sample_size: {}", alice_up_sample_size);
    SPDLOG_INFO("intersection_size: {}", intersection_size);
  } else if (Rank.getValue() == 1) {
    yacl::Buffer self_count_buffer =
        spu::psi::utils::SerializeSize(items.size());
    link_ctx->SendAsync(link_ctx->NextRank(), self_count_buffer,
                        fmt::format("send items count: {}", items.size()));

    spu::psi::DpPsiOptions options = GetDpPsiOptions(items.size());

    size_t bob_sub_sample_size;
    std::vector<size_t> intersection_idx = spu::psi::RunDpEcdhPsiBob(
        options, link_ctx, items, &bob_sub_sample_size);

    SPDLOG_INFO("bob_sub_sample_size: {}", bob_sub_sample_size);
    SPDLOG_INFO("intersection_idx size: {}", intersection_idx.size());

    WriteCsvData(OutPathOpt.getValue(), items, intersection_idx);
  } else {
    SPU_THROW("wrong rank");
  }

  return 0;
}
