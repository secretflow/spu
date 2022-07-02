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

#include "spu/psi/cryptor/ecdh_oprf/basic_ecdh_oprf.h"

#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"

#include "spu/psi/cryptor/ecdh_oprf/ecdh_oprf_selector.h"

namespace spu {
struct TestParams {
  size_t items_size;
  CurveType type = CurveType::CurveSecp256k1;
};

class BasicEcdhOprfTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(BasicEcdhOprfTest, Works) {
  auto params = GetParam();
  std::random_device rd;
  yasl::PseudoRandomGenerator<uint64_t> prg(rd());

  std::shared_ptr<IEcdhOprfServer> dh_oprf_server =
      CreateEcdhOprfServer(OprfType::Basic, params.type);

  std::shared_ptr<IEcdhOprfClient> dh_oprf_client =
      CreateEcdhOprfClient(OprfType::Basic, params.type);

  std::vector<std::string> items_vec(params.items_size);

  for (size_t idx = 0; idx < params.items_size; ++idx) {
    items_vec[idx].resize(kEccKeySize);
    prg.Fill(absl::MakeSpan(items_vec[idx]));
  }

  std::string server_evaluted = dh_oprf_server->FullEvaluate(items_vec[0]);

  std::string blinded_item = dh_oprf_client->Blind(items_vec[0]);
  std::string mask_item = dh_oprf_server->Evaluate(blinded_item);
  std::string client_evaluted =
      dh_oprf_client->Finalize(items_vec[0], mask_item);

  EXPECT_EQ(server_evaluted, client_evaluted);

  std::vector<std::string> server_evaluted_vec =
      dh_oprf_server->FullEvaluate(items_vec);
  std::vector<std::string> blinded_item_vec = dh_oprf_client->Blind(items_vec);

  std::vector<std::string> mask_item_vec =
      dh_oprf_server->Evaluate(blinded_item_vec);

  std::vector<std::string> client_evaluted_vec =
      dh_oprf_client->Finalize(items_vec, mask_item_vec);

  EXPECT_EQ(server_evaluted_vec, client_evaluted_vec);
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, BasicEcdhOprfTest,
    testing::Values(TestParams{1}, TestParams{10}, TestParams{50},
                    // fourq
                    TestParams{1, CurveType::CurveFourQ},
                    TestParams{10, CurveType::CurveFourQ},
                    TestParams{50, CurveType::CurveFourQ}));
}  // namespace spu