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

#include "libspu/mpc/cheetah/ot/ferret.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class FerretCOTTest : public testing::TestWithParam<FieldType> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, FerretCOTTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<FerretCOTTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(FerretCOTTest, ChosenCorrelationChosenChoice) {
  size_t kWorldSize = 2;
  size_t n = 10;
  auto field = GetParam();

  auto _correlation = ring_rand(field, n);
  std::vector<uint8_t> choices(n);
  std::default_random_engine rdv;
  std::uniform_int_distribution<uint64_t> uniform(0, -1);
  std::generate_n(choices.begin(), n, [&]() -> uint8_t {
    return static_cast<uint8_t>(uniform(rdv) & 1);
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto correlation = xt_adapt<ring2k_t>(_correlation);
    std::vector<ring2k_t> computed[2];
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      int rank = ctx->Rank();
      computed[rank].resize(n);
      FerretOT ferret(conn, rank == 0);
      if (rank == 0) {
        ferret.SendCAMCC({correlation.data(), correlation.size()},
                         absl::MakeSpan(computed[0]));
        ferret.Flush();
      } else {
        ferret.RecvCAMCC(absl::MakeSpan(choices), absl::MakeSpan(computed[1]));
      }
    });

    for (size_t i = 0; i < n; ++i) {
      ring2k_t c = -computed[0][i] + computed[1][i];
      ring2k_t e = choices[i] ? correlation[i] : 0;
      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(FerretCOTTest, RndMsgRndChoice) {
  size_t kWorldSize = 2;
  auto field = GetParam();
  constexpr size_t bw = 2;

  size_t n = 10;
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    std::vector<ring2k_t> msg0(n);
    std::vector<ring2k_t> msg1(n);
    ring2k_t max = static_cast<ring2k_t>(1) << bw;

    std::vector<uint8_t> choices(n);
    std::vector<ring2k_t> selected(n);

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      int rank = ctx->Rank();
      FerretOT ferret(conn, rank == 0);
      if (rank == 0) {
        ferret.SendRMRC(absl::MakeSpan(msg0), absl::MakeSpan(msg1), bw);
        ferret.Flush();
      } else {
        ferret.RecvRMRC(absl::MakeSpan(choices), absl::MakeSpan(selected), bw);
      }
    });

    for (size_t i = 0; i < n; ++i) {
      ring2k_t e = choices[i] ? msg1[i] : msg0[i];
      ring2k_t c = selected[i];
      EXPECT_TRUE(choices[i] < 2);
      EXPECT_LT(e, max);
      EXPECT_LT(c, max);
      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(FerretCOTTest, ChosenMsgChosenChoice) {
  size_t kWorldSize = 2;
  size_t n = 106;
  auto field = GetParam();
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using scalar_t = ring2k_t;
    std::default_random_engine rdv;
    std::uniform_int_distribution<uint32_t> uniform(0, -1);
    for (size_t bw : {2UL, 4UL, sizeof(scalar_t) * 8}) {
      scalar_t mask = (static_cast<scalar_t>(1) << bw) - 1;
      for (size_t N : {2, 3, 8}) {
        auto _msg = ring_rand(field, N * n);
        auto msg = xt_mutable_adapt<scalar_t>(_msg);
        msg &= mask;
        std::vector<uint8_t> choices(n);

        std::generate_n(choices.begin(), n, [&]() -> uint8_t {
          return static_cast<uint8_t>(uniform(rdv) % N);
        });

        std::vector<scalar_t> selected(n);

        utils::simulate(kWorldSize,
                        [&](std::shared_ptr<yacl::link::Context> ctx) {
                          auto conn = std::make_shared<Communicator>(ctx);
                          int rank = ctx->Rank();
                          FerretOT ferret(conn, rank == 0);
                          if (rank == 0) {
                            ferret.SendCMCC({msg.data(), msg.size()}, N, bw);
                            ferret.Flush();
                          } else {
                            ferret.RecvCMCC(absl::MakeSpan(choices), N,
                                            absl::MakeSpan(selected), bw);
                          }
                        });

        for (size_t i = 0; i < n; ++i) {
          scalar_t e = msg[i * N + choices[i]];
          scalar_t c = selected[i];
          EXPECT_EQ(e, c);
        }
      }
    }
  });
}

}  // namespace spu::mpc::cheetah::test
