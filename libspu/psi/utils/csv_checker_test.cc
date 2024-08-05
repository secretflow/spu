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

#include "libspu/psi/utils/csv_checker.h"

#include <filesystem>

#include "gtest/gtest.h"

#include "libspu/psi/io/io.h"

namespace spu::psi {

namespace {
struct TestParams {
  std::string content_a;
  std::string content_b;
  std::vector<std::string> ids_a;
  std::vector<std::string> ids_b;
  uint32_t item_size_a;
  uint32_t item_size_b;
  bool skip_check = false;
};

struct FailedTestParams {
  std::string content;
  std::vector<std::string> ids;
};
}  // namespace

class CsvCheckerTest : public testing::TestWithParam<TestParams> {
 protected:
  void SetUp() override {
    tmp_file_path_a_ = "csv_checker_test_file_a";
    tmp_file_path_b_ = "csv_checker_test_file_b";
  }
  void TearDown() override {
    if (!tmp_file_path_a_.empty()) {
      std::error_code ec;
      std::filesystem::remove(tmp_file_path_a_, ec);
      // Leave error as it is, do nothing
    }
    if (!tmp_file_path_b_.empty()) {
      std::error_code ec;
      std::filesystem::remove(tmp_file_path_b_, ec);
      // Leave error as it is, do nothing
    }
  }

  std::string tmp_file_path_a_;
  std::string tmp_file_path_b_;
};

TEST_P(CsvCheckerTest, Works) {
  auto params = GetParam();

  auto a_os = io::BuildOutputStream(io::FileIoOptions(tmp_file_path_a_));
  a_os->Write(params.content_a);
  a_os->Close();

  auto b_os = io::BuildOutputStream(io::FileIoOptions(tmp_file_path_b_));
  b_os->Write(params.content_b);
  b_os->Close();

  CsvChecker checker_a(tmp_file_path_a_, params.ids_a, "./", params.skip_check);
  CsvChecker checker_b(tmp_file_path_b_, params.ids_b, "./", params.skip_check);

  ASSERT_EQ(params.item_size_a, checker_a.data_count());
  ASSERT_EQ(params.item_size_b, checker_b.data_count());
}

class CsvCheckerDigestEqualTest : public testing::TestWithParam<TestParams> {
 protected:
  void SetUp() override {
    tmp_file_path_a_ = "csv_checker_test_file_a";
    tmp_file_path_b_ = "csv_checker_test_file_b";
  }
  void TearDown() override {
    if (!tmp_file_path_a_.empty()) {
      std::error_code ec;
      std::filesystem::remove(tmp_file_path_a_, ec);
      // Leave error as it is, do nothing
    }
    if (!tmp_file_path_b_.empty()) {
      std::error_code ec;
      std::filesystem::remove(tmp_file_path_b_, ec);
      // Leave error as it is, do nothing
    }
  }

  std::string tmp_file_path_a_;
  std::string tmp_file_path_b_;
};

TEST_P(CsvCheckerDigestEqualTest, HashDigestEqual) {
  auto params = GetParam();

  auto a_os = io::BuildOutputStream(io::FileIoOptions(tmp_file_path_a_));
  std::cout << "content_a:" << params.content_a << std::endl;
  a_os->Write(params.content_a);
  a_os->Close();

  auto b_os = io::BuildOutputStream(io::FileIoOptions(tmp_file_path_b_));
  std::cout << "content_b:" << params.content_b << std::endl;
  b_os->Write(params.content_b);
  b_os->Close();

  CsvChecker checker_a(tmp_file_path_a_, params.ids_a, "./", params.skip_check);
  CsvChecker checker_b(tmp_file_path_b_, params.ids_b, "./", params.skip_check);

  ASSERT_EQ(params.item_size_a, checker_a.data_count());
  ASSERT_EQ(params.item_size_b, checker_b.data_count());
  ASSERT_EQ(checker_a.hash_digest(), checker_b.hash_digest());
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, CsvCheckerTest,
                         testing::Values(TestParams{"id\nc\nb\na\n",
                                                    "x1,x2,id\n,,a\n,,b\n,,c\n",
                                                    {"id"},
                                                    {"id"},
                                                    3,
                                                    3},
                                         TestParams{"x1,id\n1,a\n2,b\n3,c\n",
                                                    "x1,id\n2,b\n1,a\n3,c\n",
                                                    {"id", "x1"},
                                                    {"id", "x1"},
                                                    3,
                                                    3},
                                         TestParams{"x1,id\n1,a\n2,b\n3,c\n",
                                                    "x1,id\n2,b\n1,a\n3,c\n",
                                                    {"id", "x1"},
                                                    {"id", "x1"},
                                                    3,
                                                    3,
                                                    true}));

INSTANTIATE_TEST_SUITE_P(
    HashDigestEqual_Instances, CsvCheckerDigestEqualTest,
    testing::Values(
        TestParams{"id\nc\nb\na\n", "id\nc\nb\na\n", {"id"}, {"id"}, 3, 3},
        TestParams{"x1,id\n1,a\n2,b\n3,c\n",
                   "x1,id\n1,a\n2,b\n3,c\n",
                   {"id", "x1"},
                   {"id", "x1"},
                   3,
                   3}));

class CsvCheckerFailedTest : public testing::TestWithParam<FailedTestParams> {
 protected:
  void SetUp() override { tmp_file_path_ = "csv_checker_failed_test_file"; }
  void TearDown() override {
    if (!tmp_file_path_.empty()) {
      std::error_code ec;
      std::filesystem::remove(tmp_file_path_, ec);
      // Leave error as it is, do nothing
    }
  }

  std::string tmp_file_path_;
};

TEST_P(CsvCheckerFailedTest, FailedWorks) {
  auto params = GetParam();

  auto os = io::BuildOutputStream(io::FileIoOptions(tmp_file_path_));
  os->Write(params.content);
  os->Close();

  ASSERT_ANY_THROW(CsvChecker checker(tmp_file_path_, params.ids, "./", false));
}

INSTANTIATE_TEST_SUITE_P(
    FailedWorks_Instances, CsvCheckerFailedTest,
    testing::Values(
        // ecdh
        FailedTestParams{"id\nc\nb\na\n", {"i"}},
        FailedTestParams{"x1,x2,id\n1,,a\n,2,b\n,,c\n", {"x1"}},
        FailedTestParams{"id\nc\nb\na\nc\n", {"id"}},
        FailedTestParams{"x1,id\n1,a\n2,b\n3,c\n1,a\n", {"x1, id"}}));

}  // namespace spu::psi
