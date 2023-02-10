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

#include "libspu/psi/core/bc22_psi/emp_vole.h"

#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/utils/serialize.h"

namespace spu::psi {

class EmpVoleTest : public testing::TestWithParam<size_t> {};

TEST_P(EmpVoleTest, Works) {
  auto params = GetParam();

  auto ctxs = yacl::link::test::SetupWorld(2);

  uint64_t vole_need = params;

  std::vector<WolverineVoleFieldType> vole_alice;
  std::vector<WolverineVoleFieldType> vole_bob;

  WolverineVoleFieldType delta = 0;

  std::future<void> vole_alice_thread = std::async([&] {
    WolverineVole vole(PsiRoleType::Sender, ctxs[0]);

    vole_alice = vole.Extend(vole_need);
    delta = vole.Delta();
  });

  std::future<void> vole_bob_thread = std::async([&] {
    WolverineVole vole(PsiRoleType::Receiver, ctxs[1]);

    vole_bob = vole.Extend(vole_need);
  });

  vole_alice_thread.get();
  vole_bob_thread.get();

  EXPECT_EQ(vole_alice.size(), vole_need);
  EXPECT_EQ(vole_bob.size(), vole_need);

  auto stats0 = ctxs[0]->GetStats();
  auto stats1 = ctxs[1]->GetStats();
  SPDLOG_INFO("sender/alice ctx0 sent_bytes:{} recv_bytes:{}",
              stats0->sent_bytes, stats0->recv_bytes);
  SPDLOG_INFO("receiver/bob ctx1 sent_bytes:{} recv_bytes:{}",
              stats1->sent_bytes, stats1->recv_bytes);

  // check vole
  // wi = delta * ui + vi
  // sender/alice : delta wi
  // receiver/bob : ui || vi as one __uint128_t
  for (size_t i = 0; i < vole_need; ++i) {
    // delta * ui
    WolverineVoleFieldType tmp = mod(delta * (vole_bob[i] >> 64), pr);
    // delta * ui + vi
    tmp = mod(tmp + vole_alice[i], pr);
    // check wi = delta * ui + vi
    EXPECT_EQ(tmp, (vole_bob[i] & 0xFFFFFFFFFFFFFFFFLL));
  }
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, EmpVoleTest,
                         testing::Values(10000, 100000, 1000000));

TEST(EmpVoleTest, PolynomialTest) {
  std::mt19937 rng(yacl::crypto::SecureRandU64());

  for (size_t idx = 1; idx < 4; ++idx) {
    std::vector<std::string> points(idx);

    for (auto& point : points) {
      point = std::to_string(rng());
    }

    std::vector<__uint128_t> coeffs = GetPolynomialCoefficients(points);

    for (auto& point : points) {
      WolverineVoleFieldType result =
          EvaluatePolynomial(absl::MakeSpan(coeffs), point);
      WolverineVoleFieldType tt;
      memcpy(&tt, &result, sizeof(tt));

      EXPECT_EQ(result, 0);
    }
  }
}

}  // namespace spu::psi
