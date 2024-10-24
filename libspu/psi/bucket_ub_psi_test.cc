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
#include "libspu/psi/utils/test_utils.h"

namespace spu::psi {

namespace {

std::pair<std::vector<std::string>, std::vector<std::string>>
GenerateUbPsiTestData(size_t begin, size_t items_size) {
  std::vector<std::string> items_a = test::CreateRangeItems(begin, items_size);

  std::vector<std::string> items_b = test::CreateRangeItems(
      begin + items_size / 4, std::max(static_cast<size_t>(1), items_size / 2));

  return std::make_pair(items_a, items_b);
}

void WriteCsvFile(const std::string &file_name, const std::string &column_id,
                  const std::vector<std::string> &items) {
  auto out = io::BuildOutputStream(io::FileIoOptions(file_name));
  out->Write(fmt::format("{}\n", column_id));
  for (const auto &data : items) {
    out->Write(fmt::format("{}\n", data));
  }
  out->Close();
}

void WriteSecretKey(const std::string &ecdh_secret_key_path) {
  std::ofstream wf(ecdh_secret_key_path, std::ios::out | std::ios::binary);

  std::string secret_key_binary = absl::HexStringToBytes(
      "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000");
  wf.write(secret_key_binary.data(), secret_key_binary.length());
  wf.close();
}

}  // namespace

struct TestParams {
  size_t items_size;
  CurveType curve_type = CurveType::CURVE_FOURQ;
  bool shuffle_online = false;
};

class UnbalancedPsiTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(UnbalancedPsiTest, Work) {
  auto params = GetParam();

  std::string tmp_dir = fmt::format("tmp_{}", yacl::crypto::SecureRandU64());
  std::filesystem::create_directory(tmp_dir);

  // register remove of temp dir.
  ON_SCOPE_EXIT([&] {
    if (!tmp_dir.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(tmp_dir, ec);
      if (ec.value() != 0) {
        SPDLOG_WARN("can not remove tmp dir: {}, msg: {}", tmp_dir,
                    ec.message());
      }
    }
  });

  auto lctxs = yacl::link::test::SetupWorld(2);

  std::vector<std::string> input_paths(lctxs[0]->WorldSize());
  std::vector<std::string> output_paths(lctxs[0]->WorldSize());

  std::vector<std::vector<std::string>> items_data(lctxs[0]->WorldSize());

  std::tie(items_data[0], items_data[1]) =
      GenerateUbPsiTestData(0, params.items_size);

  std::vector<std::string> intersection_check =
      test::GetIntersection(items_data[0], items_data[1]);

  std::string column_id = "id";
  std::vector<std::string> field_names = {column_id};

  for (size_t i = 0; i < lctxs[0]->WorldSize(); ++i) {
    input_paths[i] = fmt::format("{}/ub-input-rank-{}.csv", tmp_dir, i);

    output_paths[i] = fmt::format("{}/ub-output-rank-{}.csv", tmp_dir, i);
    WriteCsvFile(input_paths[i], column_id, items_data[i]);
  }

  size_t server_rank = 0;
  size_t client_rank = 1;

  std::string preprocess_file_path =
      fmt::format("{}/preprocess-cipher-store-{}.csv", tmp_dir, client_rank);
  std::string server_cache_path =
      fmt::format("{}/server-cache-{}.bin", tmp_dir, server_rank);
  std::string ecdh_secret_key_path =
      fmt::format("{}/ecdh-secret-key.bin", tmp_dir, server_rank);

  // write temp secret key
  WriteSecretKey(ecdh_secret_key_path);

  // step 1, gen cache
  {
    spu::psi::BucketPsiConfig config;

    config.mutable_input_params()->set_path(input_paths[server_rank]);
    config.mutable_input_params()->mutable_select_fields()->Add(
        field_names.begin(), field_names.end());
    config.mutable_input_params()->set_precheck(false);
    config.mutable_output_params()->set_path(server_cache_path);
    config.mutable_output_params()->set_need_sort(false);
    config.set_psi_type(spu::psi::PsiType::ECDH_OPRF_UB_PSI_2PC_GEN_CACHE);
    config.set_receiver_rank(server_rank);
    config.set_broadcast_result(false);
    config.set_bucket_size(10000000);
    config.set_curve_type(params.curve_type);
    config.set_ecdh_secret_key_path(ecdh_secret_key_path);

    BucketPsi ctx(config, nullptr);

    ctx.Run();
  }

  // step 2, transfer cache
  auto transfer_cache_proc = [&](int idx) -> spu::psi::PsiResultReport {
    spu::psi::BucketPsiConfig config;

    config.mutable_input_params()->set_path(input_paths[idx]);
    config.mutable_input_params()->mutable_select_fields()->Add(
        field_names.begin(), field_names.end());
    config.mutable_input_params()->set_precheck(false);
    config.mutable_output_params()->set_path(output_paths[idx]);
    config.mutable_output_params()->set_need_sort(false);
    config.set_psi_type(spu::psi::PsiType::ECDH_OPRF_UB_PSI_2PC_TRANSFER_CACHE);
    config.set_receiver_rank(client_rank);
    config.set_broadcast_result(false);
    // set min bucket size for test
    config.set_bucket_size(10000000);
    config.set_curve_type(params.curve_type);
    if (client_rank == lctxs[idx]->Rank()) {
      config.set_preprocess_path(preprocess_file_path);
    } else {
      config.set_ecdh_secret_key_path(ecdh_secret_key_path);
      config.mutable_input_params()->set_path(server_cache_path);
    }

    BucketPsi ctx(config, lctxs[idx]);
    return ctx.Run();
  };

  size_t world_size = lctxs.size();
  std::vector<std::future<spu::psi::PsiResultReport>> transfer_cache_f(
      world_size);
  for (size_t i = 0; i < world_size; i++) {
    transfer_cache_f[i] = std::async(transfer_cache_proc, i);
  }

  for (size_t i = 0; i < world_size; i++) {
    auto report = transfer_cache_f[i].get();
    SPDLOG_INFO("{}", report.intersection_count());
    if (i != client_rank) {
      EXPECT_EQ(report.original_count(), items_data[i].size());
    }
  }

  size_t receiver_rank = client_rank;
  spu::psi::PsiType psi_protocol =
      spu::psi::PsiType::ECDH_OPRF_UB_PSI_2PC_ONLINE;
  if (params.shuffle_online) {
    psi_protocol = spu::psi::PsiType::ECDH_OPRF_UB_PSI_2PC_SHUFFLE_ONLINE;
    receiver_rank = server_rank;
  }
  // online phase
  auto online_proc = [&](int idx) -> spu::psi::PsiResultReport {
    std::string input_file_path = input_paths[idx];
    std::string output_file_path = output_paths[idx];

    spu::psi::BucketPsiConfig config;
    config.mutable_input_params()->set_path(input_file_path);
    config.mutable_input_params()->mutable_select_fields()->Add(
        field_names.begin(), field_names.end());
    config.mutable_input_params()->set_precheck(false);
    config.mutable_output_params()->set_path(output_file_path);
    config.mutable_output_params()->set_need_sort(false);
    config.set_psi_type(psi_protocol);
    config.set_receiver_rank(receiver_rank);
    config.set_broadcast_result(false);
    // set min bucket size for test
    config.set_bucket_size(10000000);
    config.set_curve_type(params.curve_type);

    if (client_rank == lctxs[idx]->Rank()) {
      config.set_preprocess_path(preprocess_file_path);
    } else {
      config.set_preprocess_path(server_cache_path);
      config.set_ecdh_secret_key_path(ecdh_secret_key_path);
    }

    BucketPsi ctx(config, lctxs[idx]);
    return ctx.Run();
  };

  std::vector<std::future<spu::psi::PsiResultReport>> online_f(world_size);
  for (size_t i = 0; i < world_size; i++) {
    online_f[i] = std::async(online_proc, i);
  }

  for (size_t i = 0; i < world_size; i++) {
    auto report = online_f[i].get();
    SPDLOG_INFO("{}", report.intersection_count());

    if (params.shuffle_online) {
      if (i == server_rank) {
        EXPECT_EQ(report.original_count(), items_data[i].size());
        EXPECT_EQ(report.intersection_count(), intersection_check.size());
      }
    } else {
      if (i == client_rank) {
        EXPECT_EQ(report.original_count(), items_data[i].size());
        EXPECT_EQ(report.intersection_count(), intersection_check.size());
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, UnbalancedPsiTest,
    testing::Values(
        // CURVE_FOURQ
        TestParams{1},                                    //
        TestParams{10},                                   //
        TestParams{50},                                   //
        TestParams{4095},                                 // less than one batch
        TestParams{4096},                                 // exactly one batch
        TestParams{10000},                                // more than one batch
        TestParams{10000, CurveType::CURVE_FOURQ, true},  // shuffle online
        TestParams{50000, CurveType::CURVE_FOURQ, true},  // shuffle online
        TestParams{1000, CurveType::CURVE_SM2},           // CURVE_SM2
        TestParams{1000, CurveType::CURVE_SECP256K1}      // Curve256k1
        )                                                 //
);

}  // namespace spu::psi
