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

#include "libspu/pir/seal_mpir.h"

#include <chrono>
#include <random>
#include <set>

#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/cryptor/sodium_curve25519_cryptor.h"

namespace spu::pir {
namespace {
struct TestParams {
  size_t batch_number;
  size_t element_number;
  size_t element_size = 288;
  size_t poly_degree = 8192;  // now only support 8192
};

std::vector<uint8_t> GenerateDbData(TestParams params) {
  std::vector<uint8_t> db_data(params.element_number * params.element_size);

  std::random_device rd;

  std::mt19937 gen(rd());

  for (uint64_t i = 0; i < params.element_number; i++) {
    for (uint64_t j = 0; j < params.element_size; j++) {
      auto val = gen() % 256;
      db_data[(i * params.element_size) + j] = val;
    }
  }
  return db_data;
}

std::vector<size_t> GenerateQueryIndex(size_t batch_number,
                                       size_t element_number) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::set<size_t> query_index_set;

  while (true) {
    query_index_set.insert(gen() % element_number);

    if (query_index_set.size() == batch_number) {
      break;
    }
  }

  std::vector<size_t> query_index;
  query_index.assign(query_index_set.begin(), query_index_set.end());
  return query_index;
}

using DurationMillis = std::chrono::duration<double, std::milli>;
}  // namespace

class SealMultiPirTest : public testing::TestWithParam<TestParams> {};

TEST_P(SealMultiPirTest, Works) {
  auto params = GetParam();
  size_t element_number = params.element_number;
  size_t element_size = params.element_size;
  size_t batch_number = params.batch_number;
  // size_t batch_number = 256;
  double factor = 1.5;
  size_t hash_num = 3;
  spu::psi::CuckooIndex::Options cuckoo_params{batch_number, 0, hash_num,
                                               factor};

  std::vector<size_t> query_index =
      GenerateQueryIndex(batch_number, element_number);

  std::vector<uint8_t> db_bytes = GenerateDbData(params);

  auto ctxs = yacl::link::test::SetupWorld(2);

  // use dh key exchange get shared oracle seed
  psi::SodiumCurve25519Cryptor c25519_cryptor0;
  psi::SodiumCurve25519Cryptor c25519_cryptor1;

  std::future<std::vector<uint8_t>> ke_func_server =
      std::async([&] { return c25519_cryptor0.KeyExchange(ctxs[0]); });
  std::future<std::vector<uint8_t>> ke_func_client =
      std::async([&] { return c25519_cryptor1.KeyExchange(ctxs[1]); });

  std::vector<uint8_t> seed_server = ke_func_server.get();
  std::vector<uint8_t> seed_client = ke_func_client.get();

  EXPECT_EQ(seed_server, seed_client);

  spu::pir::MultiQueryOptions options{
      {params.poly_degree, element_number, element_size}, batch_number};

  SPDLOG_INFO("element_number:{}", options.seal_options.element_number);

  spu::pir::MultiQueryServer mpir_server(options, cuckoo_params, seed_server);

  spu::pir::MultiQueryClient mpir_client(options, cuckoo_params, seed_client);

  // server setup data
  mpir_server.SetDatabase(db_bytes);

  // online send galoiskey(28MB) cause: pipeline check unittest timeout
  /*
    // client send galois keys to server
    std::future<void> client_galkey_func =
        std::async([&] { return mpir_client.SendGaloisKeys(ctxs[0]); });
    std::future<void> server_galkey_func =
        std::async([&] { return mpir_server.RecvGaloisKeys(ctxs[1]); });

    client_galkey_func.get();
    server_galkey_func.get();
  */

  seal::GaloisKeys galkey = mpir_client.GenerateGaloisKeys();
  mpir_server.SetGaloisKeys(galkey);

  // do pir query/answer
  std::future<void> pir_service_func =
      std::async([&] { return mpir_server.DoMultiPirAnswer(ctxs[0]); });
  std::future<std::vector<std::vector<uint8_t>>> pir_client_func = std::async(
      [&] { return mpir_client.DoMultiPirQuery(ctxs[1], query_index); });

  pir_service_func.get();
  std::vector<std::vector<uint8_t>> query_reply_bytes = pir_client_func.get();

  EXPECT_EQ(query_reply_bytes.size(), query_index.size());

  for (size_t idx = 0; idx < query_reply_bytes.size(); ++idx) {
    std::vector<uint8_t> query_db_bytes(params.element_size);
    std::memcpy(query_db_bytes.data(),
                &db_bytes[query_index[idx] * params.element_size],
                params.element_size);

    if ((query_db_bytes.size() != query_reply_bytes[idx].size()) ||
        (std::memcmp(query_db_bytes.data(), query_reply_bytes[idx].data(),
                     query_reply_bytes[idx].size()) != 0)) {
      SPDLOG_INFO(
          "idx:{} query_index:{} query_db_bytes:{}", idx, query_index[idx],
          absl::BytesToHexString(absl::string_view(
              (const char *)query_db_bytes.data(), query_db_bytes.size())));
      SPDLOG_INFO("query_reply_bytes[{}]:{}", idx,
                  absl::BytesToHexString(absl::string_view(
                      (const char *)query_reply_bytes[idx].data(),
                      query_reply_bytes[idx].size())));
    }

    EXPECT_EQ(query_db_bytes, query_reply_bytes[idx]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, SealMultiPirTest,
    testing::Values(TestParams{32, 1000},       // element size default 288B
                    TestParams{32, 1000, 10},   //
                    TestParams{32, 1000, 400},  //
                    TestParams{64, 10000},      // element size default 288B
                    TestParams{64, 10000, 20})  //
);

}  // namespace spu::pir
