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

#include "spu/psi/bucket_psi.h"

#include <fstream>
#include <vector>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yasl/link/test_util.h"

#include "spu/psi/io/io.h"

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

void WritFile(const std::string& file_name, const std::string& content) {
  auto out = io::BuildOutputStream(io::FileIoOptions(file_name));
  out->Write(content);
  out->Close();
}

}  // namespace

class StreamTaskPsiTest : public testing::TestWithParam<TestParams> {
 protected:
  void SetUp() override {
    tmp_dir_ = "./tmp";
    std::filesystem::create_directory(tmp_dir_);
  }
  void TearDown() override {
    input_paths_.clear();
    output_paths_.clear();

    if (!tmp_dir_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(tmp_dir_, ec);
      // Leave error as it is, do nothing
    }
  }

  void SetupTmpfilePaths(size_t num) {
    for (size_t i = 0; i < num; ++i) {
      input_paths_.emplace_back(fmt::format("{}/tmp-input-{}", tmp_dir_, i));
      output_paths_.emplace_back(fmt::format("{}/tmp-output-{}", tmp_dir_, i));
    }
  }

 protected:
  std::string tmp_dir_;
  std::vector<std::string> input_paths_;
  std::vector<std::string> output_paths_;
};

TEST_P(StreamTaskPsiTest, Works) {
  auto params = GetParam();

  SetupTmpfilePaths(params.in_content_list.size());
  auto lctxs = yasl::link::test::SetupWorld(params.in_content_list.size());

  auto proc = [&](int idx) -> spu::psi::PsiResultReport {
    spu::psi::BucketPsiConfig config;
    config.mutable_input_params()->set_path(input_paths_[idx]);
    config.mutable_input_params()->mutable_select_fields()->Add(
        params.field_names_list[idx].begin(),
        params.field_names_list[idx].end());
    config.mutable_input_params()->set_precheck(true);
    config.mutable_output_params()->set_path(output_paths_[idx]);
    config.mutable_output_params()->set_need_sort(params.should_sort);
    config.set_psi_type(params.psi_protocol);
    config.set_broadcast_result(true);
    // set small bucket size for test
    config.set_bucket_size(3);

    BucketPsi ctx(config, lctxs[idx]);
    return ctx.Run();
  };

  size_t world_size = lctxs.size();
  std::vector<std::future<spu::psi::PsiResultReport>> f_links(world_size);
  for (size_t i = 0; i < world_size; i++) {
    WritFile(input_paths_[i], params.in_content_list[i]);
    f_links[i] = std::async(proc, i);
  }

  for (size_t i = 0; i < world_size; i++) {
    auto report = f_links[i].get();

    EXPECT_EQ(report.original_count(), params.item_size_list[i]);
    EXPECT_EQ(report.intersection_count(),
              GetFileLineCount(output_paths_[i]) - 1);
    EXPECT_EQ(params.expect_result_size,
              GetFileLineCount(output_paths_[i]) - 1);
    EXPECT_EQ(params.out_content_list[i], ReadFileToString(output_paths_[i]));
  }
}

TEST_P(StreamTaskPsiTest, BroadcastFalse) {
  auto params = GetParam();
  size_t receiver_rank = 0;

  SetupTmpfilePaths(params.in_content_list.size());
  auto lctxs = yasl::link::test::SetupWorld(params.in_content_list.size());

  auto proc = [&](int idx) -> spu::psi::PsiResultReport {
    spu::psi::BucketPsiConfig config;
    config.mutable_input_params()->set_path(input_paths_[idx]);
    config.mutable_input_params()->mutable_select_fields()->Add(
        params.field_names_list[idx].begin(),
        params.field_names_list[idx].end());
    config.mutable_input_params()->set_precheck(true);
    config.mutable_output_params()->set_path(output_paths_[idx]);
    config.mutable_output_params()->set_need_sort(params.should_sort);
    config.set_psi_type(params.psi_protocol);
    config.set_receiver_rank(receiver_rank);
    config.set_broadcast_result(false);
    // set min bucket size for test
    config.set_bucket_size(1);

    BucketPsi ctx(config, lctxs[idx]);
    return ctx.Run();
  };

  size_t world_size = lctxs.size();
  std::vector<std::future<spu::psi::PsiResultReport>> f_links(world_size);
  for (size_t i = 0; i < world_size; i++) {
    WritFile(input_paths_[i], params.in_content_list[i]);
    f_links[i] = std::async(proc, i);
  }

  for (size_t i = 0; i < world_size; i++) {
    auto report = f_links[i].get();

    if (i == receiver_rank) {
      EXPECT_EQ(report.original_count(), params.item_size_list[i]);
      EXPECT_EQ(report.intersection_count(),
                GetFileLineCount(output_paths_[i]) - 1);
      EXPECT_EQ(params.expect_result_size,
                GetFileLineCount(output_paths_[i]) - 1);
      EXPECT_EQ(params.out_content_list[i], ReadFileToString(output_paths_[i]));
    } else {
      EXPECT_EQ(report.original_count(), params.item_size_list[i]);
      EXPECT_EQ(report.intersection_count(), -1);
      EXPECT_FALSE(std::filesystem::exists(output_paths_[i]));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, StreamTaskPsiTest,
    testing::Values(
        TestParams{
            {10, 15},
            {"id,value\ng,1\nb,2\ns,1\nh,1\ne,1\nc,1\na,1\nj,1\nk,1\nl,1\n",
             "id,value\ne,1\nc,1\na,1\nj,1\nk,1\nq,1\nw,1\nn,1\nr,1\nt,1\ny,"
             "1\nu,1\ni,1\no,1\np,1\n"},
            {"id,value\na,1\nc,1\ne,1\nj,1\nk,1\n",
             "id,value\na,1\nc,1\ne,1\nj,1\nk,1\n"},
            {{"id"}, {"id"}},
            spu::psi::PsiType::ECDH_PSI_2PC,
            64,
            true,
            5,
        },
        TestParams{
            {3, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n",
             "id,value\nb测试,b\nc测试,c\na测试,a\n"},
            {"id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n"},
            {{"id"}, {"id"}},
            spu::psi::PsiType::ECDH_PSI_2PC,
            64,
            true,
            3,
        },
        TestParams{
            {3, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n",
             "id,value\nb测试,b\nc测试,c\na测试,a\n"},
            {"id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n"},
            {{"id"}, {"id"}},
            spu::psi::PsiType::KKRT_PSI_2PC,
            64,
            true,
            3,
        },
        TestParams{
            {5, 6},
            {"id,value\nc测试,c\nb测试,b\na测试,a\nn测试,n\nj测试,j\n",
             "id,value\nb测试,b\nc测试,c\nj测试,t\na测试,a\nk测试,k\nq测试,"
             "q\n"},
            {"id,value\na测试,a\nb测试,b\nc测试,c\nj测试,j\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\nj测试,t\n"},
            {{"id"}, {"id"}},
            spu::psi::PsiType::BC22_PSI_2PC,
            64,
            true,
            4,
        },
        TestParams{
            {3, 3, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n",
             "id,value\nb测试,b\nc测试,c\na测试,a\n",
             "id,value\na测试,a\nc测试,c\nb测试,b\n"},
            {"id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n"},
            {{"id"}, {"id"}, {"id"}},
            spu::psi::PsiType::ECDH_PSI_3PC,
            64,
            true,
            3,
        },
        TestParams{
            {3, 3, 3, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n",
             "id,value\nb测试,b\nc测试,c\na测试,a\n",
             "id,value\na测试,a\nc测试,c\nb测试,b\n",
             "id,value\na测试,b\nc测试,c\nb测试,b\n"},
            {"id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,b\nb测试,b\nc测试,c\n"},
            {{"id"}, {"id"}, {"id"}, {"id"}},
            spu::psi::PsiType::ECDH_PSI_NPC,
            64,
            true,
            3,
        },
        TestParams{
            {3, 3, 3, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n",
             "id,value\nb测试,b\nc测试,c\na测试,a\n",
             "id,value\na测试,a\nc测试,c\nb测试,b\n",
             "id,value\na测试,b\nc测试,c\nb测试,b\n"},
            {"id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,a\nb测试,b\nc测试,c\n",
             "id,value\na测试,b\nb测试,b\nc测试,c\n"},
            {{"id"}, {"id"}, {"id"}, {"id"}},
            spu::psi::PsiType::KKRT_PSI_NPC,
            64,
            true,
            3,
        },

        // one party empty
        TestParams{
            {3, 0},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n", "id,value\n"},
            {"id,value\n", "id,value\n"},
            {{"id"}, {"id"}},
            spu::psi::PsiType::ECDH_PSI_2PC,
            64,
            true,
            0,
        },
        TestParams{
            {3, 0},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n", "id,value\n"},
            {"id,value\n", "id,value\n"},
            {{"id"}, {"id"}},
            spu::psi::PsiType::KKRT_PSI_2PC,
            64,
            true,
            0,
        },
        TestParams{
            {3, 0, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n", "id,value\n",
             "id,value\na测试,a\nc测试,c\nb测试,b\n"},
            {"id,value\n", "id,value\n", "id,value\n"},
            {{"id"}, {"id"}, {"id"}},
            spu::psi::PsiType::ECDH_PSI_3PC,
            64,
            true,
            0,
        },
        TestParams{
            {3, 0, 3, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n", "id,value\n",
             "id,value\na测试,a\nc测试,c\nb测试,b\n",
             "id,value\na测试,b\nc测试,c\nb测试,b\n"},
            {"id,value\n", "id,value\n", "id,value\n", "id,value\n"},
            {{"id"}, {"id"}, {"id"}, {"id"}},
            spu::psi::PsiType::ECDH_PSI_NPC,
            64,
            true,
            0,
        },
        TestParams{
            {3, 0, 3, 3},
            {"id,value\nc测试,c\nb测试,b\na测试,a\n", "id,value\n",
             "id,value\na测试,a\nc测试,c\nb测试,b\n",
             "id,value\na测试,b\nc测试,c\nb测试,b\n"},
            {"id,value\n", "id,value\n", "id,value\n", "id,value\n"},
            {{"id"}, {"id"}, {"id"}, {"id"}},
            spu::psi::PsiType::KKRT_PSI_NPC,
            64,
            true,
            0,
        },

        // multi key
        TestParams{
            {3, 2},
            {"f2,id\n1,a\n1,b\n6,c\n", "f1,id\n1,b\n6,c\n"},
            {"f2,id\n1,b\n6,c\n", "f1,id\n1,b\n6,c\n"},
            {{"f2", "id"}, {"f1", "id"}},
            spu::psi::PsiType::ECDH_PSI_2PC,
            64,
            true,
            2,
        },
        TestParams{
            {3, 2},
            {"f2,id\n1,a\n1,b\n6,c\n", "f1,id\n1,b\n6,c\n"},
            {"f2,id\n1,b\n6,c\n", "f1,id\n1,b\n6,c\n"},
            {{"f2", "id"}, {"f1", "id"}},
            spu::psi::PsiType::KKRT_PSI_2PC,
            64,
            true,
            2,
        },
        TestParams{
            {3, 2, 1},
            {"f2,id\n1,a\n1,b\n6,c\n", "f1,id\n1,b\n6,c\n", "f3,id\n1,b\n"},
            {"f2,id\n1,b\n", "f1,id\n1,b\n", "f3,id\n1,b\n"},
            {{"f2", "id"}, {"f1", "id"}, {"f3", "id"}},
            spu::psi::PsiType::ECDH_PSI_3PC,
            64,
            true,
            1,
        },
        TestParams{
            {3, 2, 2, 1},
            {"f2,id\n1,a\n1,b\n6,c\n", "f1,id\n1,b\n6,c\n", "f3,id\n1,b\n1,a\n",
             "f4,id\n1,b\n"},
            {"f2,id\n1,b\n", "f1,id\n1,b\n", "f3,id\n1,b\n", "f4,id\n1,b\n"},
            {{"f2", "id"}, {"f1", "id"}, {"f3", "id"}, {"f4", "id"}},
            spu::psi::PsiType::ECDH_PSI_NPC,
            64,
            true,
            1,
        },
        TestParams{
            {3, 2, 2, 1},
            {"f2,id\n1,a\n1,b\n6,c\n", "f1,id\n1,b\n6,c\n", "f3,id\n1,b\n1,a\n",
             "f4,id\n1,b\n"},
            {"f2,id\n1,b\n", "f1,id\n1,b\n", "f3,id\n1,b\n", "f4,id\n1,b\n"},
            {{"f2", "id"}, {"f1", "id"}, {"f3", "id"}, {"f4", "id"}},
            spu::psi::PsiType::KKRT_PSI_NPC,
            64,
            true,
            1,
        }));

struct FailedTestParams {
  size_t party_num;
  size_t receiver_rank;
  spu::psi::PsiType psi_protocol;
};

class BucketTaskPsiTestFailedTest
    : public testing::TestWithParam<FailedTestParams> {};

TEST_P(BucketTaskPsiTestFailedTest, FailedWorks) {
  auto params = GetParam();

  auto lctxs = yasl::link::test::SetupWorld(params.party_num);

  spu::psi::BucketPsiConfig config;
  config.set_psi_type(params.psi_protocol);
  config.set_receiver_rank(params.receiver_rank);
  config.set_broadcast_result(true);

  ASSERT_ANY_THROW(BucketPsi ctx(config, lctxs[0]));
}

INSTANTIATE_TEST_SUITE_P(FailedWorks_Instances, BucketTaskPsiTestFailedTest,
                         testing::Values(
                             // invalid link world size
                             FailedTestParams{3, 0, PsiType::KKRT_PSI_2PC},
                             FailedTestParams{4, 0, PsiType::ECDH_PSI_2PC},
                             FailedTestParams{1, 0, PsiType::BC22_PSI_2PC},
                             FailedTestParams{2, 0, PsiType::ECDH_PSI_3PC},
                             // invalid receiver_rank
                             FailedTestParams{3, 4, PsiType::ECDH_PSI_3PC},
                             FailedTestParams{2, 5, PsiType::KKRT_PSI_2PC},
                             // invalid psi_type
                             FailedTestParams{3, 4,
                                              PsiType::INVALID_PSI_TYPE}));

}  // namespace spu::psi