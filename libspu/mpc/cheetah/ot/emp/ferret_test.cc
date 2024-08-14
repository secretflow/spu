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

#include "libspu/mpc/cheetah/ot/emp/ferret.h"

#include <random>

#include "gtest/gtest.h"

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

template <typename T>
absl::Span<T> makeSpan(NdArrayView<T> a) {
  return {&a[0], (size_t)a.numel()};
}

template <typename T>
absl::Span<const T> makeConstSpan(NdArrayView<T> a) {
  return {&a[0], (size_t)a.numel()};
}

TEST_P(FerretCOTTest, ChosenCorrelationChosenChoice) {
  size_t kWorldSize = 2;
  int64_t n = 10;
  auto field = GetParam();

  auto _correlation = ring_rand(field, {n});
  std::vector<uint8_t> choices(n);
  std::default_random_engine rdv;
  std::uniform_int_distribution<uint64_t> uniform(0, -1);
  std::generate_n(choices.begin(), n, [&]() -> uint8_t {
    return static_cast<uint8_t>(uniform(rdv) & 1);
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> correlation(_correlation);
    std::vector<ring2k_t> computed[2];
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      int rank = ctx->Rank();
      computed[rank].resize(n);
      EmpFerretOt ferret(conn, rank == 0);
      if (rank == 0) {
        ferret.SendCAMCC(makeConstSpan<ring2k_t>(correlation),
                         absl::MakeSpan(computed[0]));
        ferret.Flush();
      } else {
        ferret.RecvCAMCC(absl::MakeSpan(choices), absl::MakeSpan(computed[1]));
      }
    });

    for (int64_t i = 0; i < n; ++i) {
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
  DISPATCH_ALL_FIELDS(field, [&]() {
    std::vector<ring2k_t> msg0(n);
    std::vector<ring2k_t> msg1(n);
    ring2k_t max = static_cast<ring2k_t>(1) << bw;

    std::vector<uint8_t> choices(n);
    std::vector<ring2k_t> selected(n);

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      int rank = ctx->Rank();
      EmpFerretOt ferret(conn, rank == 0);
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

TEST_P(FerretCOTTest, RndMsgChosenChoice) {
  size_t kWorldSize = 2;
  auto field = GetParam();
  constexpr size_t bw = 2;

  size_t n = 10;
  DISPATCH_ALL_FIELDS(field, [&]() {
    std::vector<ring2k_t> msg0(n);
    std::vector<ring2k_t> msg1(n);
    ring2k_t max = static_cast<ring2k_t>(1) << bw;

    std::vector<uint8_t> choices(n);
    std::default_random_engine rdv;
    std::uniform_int_distribution<uint64_t> uniform(0, -1);
    std::generate_n(choices.begin(), n, [&]() -> uint8_t {
      return static_cast<uint8_t>(uniform(rdv) & 1);
    });

    std::vector<ring2k_t> selected(n);

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      int rank = ctx->Rank();
      EmpFerretOt ferret(conn, rank == 0);
      if (rank == 0) {
        ferret.SendRMCC(absl::MakeSpan(msg0), absl::MakeSpan(msg1), bw);
        ferret.Flush();
      } else {
        ferret.RecvRMCC(absl::MakeSpan(choices), absl::MakeSpan(selected), bw);
      }
    });

    for (size_t i = 0; i < n; ++i) {
      ring2k_t e = choices[i] ? msg1[i] : msg0[i];
      ring2k_t c = selected[i];
      EXPECT_LT(e, max);
      EXPECT_LT(c, max);
      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(FerretCOTTest, ChosenMsgChosenChoice) {
  size_t kWorldSize = 2;
  int64_t n = 100;
  auto field = GetParam();
  DISPATCH_ALL_FIELDS(field, [&]() {
    using scalar_t = ring2k_t;
    std::default_random_engine rdv;
    std::uniform_int_distribution<uint32_t> uniform(0, -1);
    for (size_t bw : {2UL, 4UL, sizeof(scalar_t) * 8}) {
      scalar_t mask = (static_cast<scalar_t>(1) << bw) - 1;
      for (int64_t N : {2, 3, 8}) {
        auto _msg = ring_rand(field, {N * n});
        NdArrayView<scalar_t> msg(_msg);
        pforeach(0, msg.numel(), [&](int64_t i) { msg[i] &= mask; });

        std::vector<uint8_t> choices(n);
        std::generate_n(choices.begin(), n, [&]() -> uint8_t {
          return static_cast<uint8_t>(uniform(rdv) % N);
        });

        std::vector<scalar_t> selected(n);

        utils::simulate(
            kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
              auto conn = std::make_shared<Communicator>(ctx);
              int rank = ctx->Rank();
              EmpFerretOt ferret(conn, rank == 0);
              if (rank == 0) {
                ferret.SendCMCC(makeConstSpan<scalar_t>(msg), N, bw);
                ferret.Flush();
              } else {
                ferret.RecvCMCC(absl::MakeSpan(choices), N,
                                absl::MakeSpan(selected), bw);
              }
            });

        for (int64_t i = 0; i < n; ++i) {
          scalar_t e = msg[i * N + choices[i]];
          scalar_t c = selected[i];
          EXPECT_EQ(e, c);
        }
      }
    }
  });
}

TEST_P(FerretCOTTest, COT_Collapse) {
  size_t kWorldSize = 2;
  int64_t n = 8;
  auto field = GetParam();

  const auto bw = SizeOf(field) * 8;
  const int level = bw;

  // generate random choices and correlation
  const auto _correlation = ring_rand(field, {static_cast<int64_t>(n * level)});
  const auto N = _correlation.numel();

  NdArrayRef oup1 = ring_zeros(field, _correlation.shape());
  NdArrayRef oup2 = ring_zeros(field, _correlation.shape());

  std::vector<uint8_t> choices(N, 1);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;

    auto out1_span = absl::MakeSpan(&oup1.at<u2k>(0), N);
    auto out2_span = absl::MakeSpan(&oup2.at<u2k>(0), N);

    NdArrayView<u2k> correlation(_correlation);

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      int rank = ctx->Rank();

      EmpFerretOt ferret(conn, rank == 0);
      if (rank == 0) {
        ferret.SendCAMCC_Collapse(makeConstSpan(correlation), out1_span, bw,
                                  level);
        ferret.Flush();

      } else {
        ferret.RecvCAMCC_Collapse(absl::MakeSpan(choices), out2_span, bw,
                                  level);
      }
    });

    // Sample-major order
    //      n ||     n     || n         || .... || n
    // k=level||k=level - 1||k=level - 2|| ....
    for (int64_t i = 0; i < N; i += n) {
      const auto cur_bw = bw - (i / n);
      const auto mask = makeMask<ring2k_t>(cur_bw);
      for (int64_t j = 0; j < n; ++j) {
        ring2k_t c = (-out1_span[i + j] + out2_span[i + j]) & mask;
        ring2k_t e = (choices[i + j] ? correlation[i + j] : 0) & mask;

        ASSERT_EQ(c, e);
      }
    }
  });
}
}  // namespace spu::mpc::cheetah::test
