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

#include "spu/pir/seal_pir.h"

#include <chrono>
#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yasl/link/test_util.h"

namespace spu::pir {
namespace {
struct TestParams {
  size_t element_number;
  size_t element_size = 288;
  size_t query_size = 0;
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

using duration_millis = std::chrono::duration<double, std::milli>;
}  // namespace

class SealPirTest : public testing::TestWithParam<TestParams> {};

TEST_P(SealPirTest, Works) {
  size_t n = 8192;
  auto params = GetParam();
  auto ctxs = yasl::link::test::SetupWorld(2);

  std::vector<uint8_t> db_data = GenerateDbData(params);

  spu::pir::SealPirOptions options{n, params.element_number,
                                   params.element_size, params.query_size};

  spu::pir::SealPirClient client(options);

  std::shared_ptr<IDbPlaintextStore> plaintext_store =
      std::make_shared<MemoryDbPlaintextStore>();
#ifdef DEC_DEBUG_
  spu::pir::SealPirServer server(options, client, plaintext_store);
#else
  spu::pir::SealPirServer server(options, plaintext_store);
#endif

  // === server setup

  std::shared_ptr<IDbElementProvider> db_provider =
      std::make_shared<MemoryDbElementProvider>(db_data, params.element_size);
  server.SetDatabase(db_provider);

  /* galkey data 28MB, cause pipeline unittest timeout
  // client send galois keys to server
  std::future<void> client_galkey_func =
      std::async([&] { return client.SendGaloisKeys(ctxs[0]); });
  std::future<void> server_galkey_func =
      std::async([&] { return server.RecvGaloisKeys(ctxs[1]); });

  client_galkey_func.get();
  server_galkey_func.get();
  */
  // use offline
  seal::GaloisKeys galkey = client.GenerateGaloisKeys();
  server.SetGaloisKeys(galkey);

  std::random_device rd;

  std::mt19937 gen(rd());
  // size_t index = 40;
  size_t index = gen() % params.element_number;

  // do pir query/answer
  const auto pir_start_time = std::chrono::system_clock::now();

  std::future<std::vector<uint8_t>> pir_client_func =
      std::async([&] { return client.DoPirQuery(ctxs[0], index); });
  std::future<void> pir_service_func =
      std::async([&] { return server.DoPirAnswer(ctxs[1]); });

  pir_service_func.get();
  std::vector<uint8_t> query_reply_bytes = pir_client_func.get();

  const auto pir_end_time = std::chrono::system_clock::now();
  const duration_millis pir_time = pir_end_time - pir_start_time;

  SPDLOG_INFO("pir time : {} ms", pir_time.count());

  EXPECT_EQ(
      std::memcmp(query_reply_bytes.data(),
                  &db_data[index * params.element_size], params.element_size),
      0);
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, SealPirTest,
    testing::Values(TestParams{1000},       // element size default 288B
                    TestParams{1000, 10},   //
                    TestParams{1000, 20},   //
                    TestParams{1000, 400},  //
                    TestParams{3000},       // element size default 288B
                    TestParams{3000, 10},   //
                    TestParams{3000, 20},   //
                    TestParams{3000, 400},  //
                    //
                    TestParams{203, 288, 100},    //
                    TestParams{3000, 288, 1000})  //
);

}  // namespace spu::pir
