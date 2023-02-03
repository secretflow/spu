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

#include <fstream>
#include <vector>

#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/bucket_psi.h"
#include "libspu/psi/io/io.h"

namespace spu::psi {

namespace {

struct TestParams {
  std::vector<uint32_t> item_size_list;
  std::vector<std::string> in_content_list;
  std::vector<std::string> out_content_list;
  std::vector<std::vector<std::string>> field_names_list;
  spu::psi::PsiType psi_protocol;
  size_t num_bins;
  bool should_sort;
  uint32_t expect_result_size;
};

size_t GetFileLineCount(const std::string& name) {
  std::ifstream in(name);
  return std::count(std::istreambuf_iterator<char>(in),
                    std::istreambuf_iterator<char>(), '\n');
}

std::string ReadFileToString(const std::string& name) {
  auto io = io::BuildInputStream(io::FileIoOptions(name));
  std::string r;
  r.resize(io->GetLength());
  io->Read(r.data(), r.size());
  return r;
}

void WriteFile(const std::string& file_name, const std::string& content) {
  auto out = io::BuildOutputStream(io::FileIoOptions(file_name));
  out->Write(content);
  out->Close();
}

}  // namespace

class UnbalancedPsiTest {
 public:
  UnbalancedPsiTest() { SetUp(); }

  ~UnbalancedPsiTest() { TearDown(); }

  void SetUp() {
    // tmp_dir_ = "./tmp";
    tmp_dir_ = fmt::format("tmp_{}", yacl::crypto::SecureRandU64());
    std::filesystem::create_directory(tmp_dir_);
  }

  void TearDown() {
    input_paths_.clear();
    output_paths_.clear();

    if (!tmp_dir_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(tmp_dir_, ec);
      //  Leave error as it is, do nothing
    }
  }

  void SetupTmpfilePaths(size_t num) {
    for (size_t i = 0; i < num; ++i) {
      input_paths_.emplace_back(fmt::format("{}/tmp-input-{}", tmp_dir_, i));
      output_paths_.emplace_back(fmt::format("{}/tmp-output-{}", tmp_dir_, i));
    }
  }

  std::string tmp_dir_;
  std::vector<std::string> input_paths_;
  std::vector<std::string> output_paths_;
};

TEST(UnbalancedPsiTest, EcdhOprfUnbalanced) {
  UnbalancedPsiTest unbalanced_psi_test;
  TestParams params = {
      {4, 15},
      {"id,value\na,1\nb,2\ne,1\nc,1\n",
       "id,value\ne,1\nc,1\na,1\nj,1\nk,1\nq,1\nw,1\nn,1\nr,1\nt,1\ny,"
       "1\nu,1\ni,1\no,1\np,1\n"},
      {"id,value\na,1\nc,1\ne,1\n", "id,value\na,1\nc,1\ne,1\n"},
      {{"id"}, {"id"}},
      spu::psi::PsiType::ECDH_OPRF_UNBALANCED_PSI_2PC_OFFLINE,
      64,
      true,
      3,
  };
  size_t receiver_rank = 0;

  unbalanced_psi_test.SetupTmpfilePaths(params.in_content_list.size());
  auto lctxs = yacl::link::test::SetupWorld(params.in_content_list.size());

  std::string preprocess_file_path =
      fmt::format("{}/preprocess-cipher-store-{}.db",
                  unbalanced_psi_test.tmp_dir_, receiver_rank);
  std::string ecdh_secret_key_path = fmt::format(
      "{}/ecdh-secret-key.bin", unbalanced_psi_test.tmp_dir_, receiver_rank);

  // write temp secret key
  {
    std::ofstream wf(ecdh_secret_key_path, std::ios::out | std::ios::binary);

    std::string secret_key_binary = absl::HexStringToBytes(
        "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000");
    wf.write(secret_key_binary.data(), secret_key_binary.length());
    wf.close();
  }

  // offline phase
  auto offline_proc = [&](int idx) -> spu::psi::PsiResultReport {
    spu::psi::BucketPsiConfig config;

    config.mutable_input_params()->set_path(
        unbalanced_psi_test.input_paths_[idx]);
    config.mutable_input_params()->mutable_select_fields()->Add(
        params.field_names_list[idx].begin(),
        params.field_names_list[idx].end());
    config.mutable_input_params()->set_precheck(true);
    config.mutable_output_params()->set_path(
        unbalanced_psi_test.output_paths_[idx]);
    config.mutable_output_params()->set_need_sort(params.should_sort);
    config.set_psi_type(params.psi_protocol);
    config.set_receiver_rank(receiver_rank);
    config.set_broadcast_result(false);
    // set min bucket size for test
    config.set_bucket_size(1);
    config.set_curve_type(CurveType::CURVE_FOURQ);
    if (receiver_rank == lctxs[idx]->Rank()) {
      config.set_preprocess_path(preprocess_file_path);
    } else {
      config.set_ecdh_secret_key_path(ecdh_secret_key_path);
    }

    BucketPsi ctx(config, lctxs[idx]);
    return ctx.Run();
  };

  size_t world_size = lctxs.size();
  std::vector<std::future<spu::psi::PsiResultReport>> offline_f_links(
      world_size);
  for (size_t i = 0; i < world_size; i++) {
    WriteFile(unbalanced_psi_test.input_paths_[i], params.in_content_list[i]);
    offline_f_links[i] = std::async(offline_proc, i);
  }

  for (size_t i = 0; i < world_size; i++) {
    auto report = offline_f_links[i].get();
    SPDLOG_INFO("{}", report.intersection_count());
    if (i != receiver_rank) {
      EXPECT_EQ(report.original_count(), params.item_size_list[i]);
    }
  }

  // online phase
  auto online_proc = [&](int idx) -> spu::psi::PsiResultReport {
    std::string input_file_path = unbalanced_psi_test.input_paths_[idx];
    std::string output_file_path = unbalanced_psi_test.output_paths_[idx];
    std::vector<std::string> id_list = params.field_names_list[idx];

    spu::psi::BucketPsiConfig config;
    config.mutable_input_params()->set_path(input_file_path);
    config.mutable_input_params()->mutable_select_fields()->Add(id_list.begin(),
                                                                id_list.end());
    config.mutable_input_params()->set_precheck(true);
    config.mutable_output_params()->set_path(output_file_path);
    config.mutable_output_params()->set_need_sort(params.should_sort);
    config.set_psi_type(spu::psi::PsiType::ECDH_OPRF_UNBALANCED_PSI_2PC_ONLINE);
    config.set_receiver_rank(receiver_rank);
    config.set_broadcast_result(false);
    // set min bucket size for test
    config.set_bucket_size(1);
    config.set_curve_type(CurveType::CURVE_FOURQ);

    if (receiver_rank == lctxs[idx]->Rank()) {
      config.set_preprocess_path(preprocess_file_path);
    } else {
      config.set_ecdh_secret_key_path(ecdh_secret_key_path);
    }

    BucketPsi ctx(config, lctxs[idx]);
    return ctx.Run();
  };

  std::vector<std::future<spu::psi::PsiResultReport>> online_f_links(
      world_size);
  for (size_t i = 0; i < world_size; i++) {
    WriteFile(unbalanced_psi_test.input_paths_[i], params.in_content_list[i]);
    online_f_links[i] = std::async(online_proc, i);
  }

  for (size_t i = 0; i < world_size; i++) {
    auto report = online_f_links[i].get();
    SPDLOG_INFO("{}", report.intersection_count());

    if (i == receiver_rank) {
      EXPECT_EQ(report.original_count(), params.item_size_list[i]);
      EXPECT_EQ(report.intersection_count(),
                GetFileLineCount(unbalanced_psi_test.output_paths_[i]) - 1);
      EXPECT_EQ(params.expect_result_size,
                GetFileLineCount(unbalanced_psi_test.output_paths_[i]) - 1);
      EXPECT_EQ(params.out_content_list[i],
                ReadFileToString(unbalanced_psi_test.output_paths_[i]));
    }
  }
}

}  // namespace spu::psi
