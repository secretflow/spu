// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/ot/yacl/ferret.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/ot/matrix_transpose.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class FerretCOTTest
    : public testing::TestWithParam<std::tuple<FieldType, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, FerretCOTTest,
    testing::Combine(testing::Values(FieldType::FM8, FieldType::FM16,
                                     FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<FerretCOTTest::ParamType> &p) {
      return fmt::format("{}SS{}", std::get<0>(p.param), std::get<1>(p.param));
    });

template <typename T>
absl::Span<const T> makeConstSpan(NdArrayView<T> a) {
  return {&a[0], (size_t)a.numel()};
}

template <typename T>
typename std::make_unsigned<T>::type makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

// cot, given delta & choice, output (x, x + delta * choice)
TEST_P(FerretCOTTest, ChosenCorrelationChosenChoice) {
  size_t kWorldSize = 2;
  int64_t n = 10;
  auto field = std::get<0>(GetParam());
  auto use_ss = std::get<1>(GetParam());

  // random correlation delta
  auto _correlation = ring_rand(field, {n});
  // generate random choices
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
      // rank = 0 -> sender
      YaclFerretOt ferret(conn, rank == 0, use_ss);
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

// random OT, random msgs, random choices
TEST_P(FerretCOTTest, RndMsgRndChoice) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto use_ss = std::get<1>(GetParam());
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
      YaclFerretOt ferret(conn, rank == 0, use_ss);
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

// random OT, random msgs, chosen choices
TEST_P(FerretCOTTest, RndMsgChosenChoice) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto use_ss = std::get<1>(GetParam());
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
      YaclFerretOt ferret(conn, rank == 0, use_ss);
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

// ot, chosen msgs, chosen choices
TEST_P(FerretCOTTest, ChosenMsgChosenChoice) {
  size_t kWorldSize = 2;
  int64_t n = 1 << 10;
  auto field = std::get<0>(GetParam());
  auto use_ss = std::get<1>(GetParam());
  DISPATCH_ALL_FIELDS(field, [&]() {
    using scalar_t = ring2k_t;
    std::default_random_engine rdv;
    std::uniform_int_distribution<uint32_t> uniform(0, -1);
    for (int64_t N : {32, 64, 128}) {
      for (size_t bw : {4UL, 8UL, 23UL, 32UL, 64UL}) {
        if (bw > SizeOf(field)) {
          continue;
        }
        scalar_t mask = (static_cast<scalar_t>(1) << bw) - 1;
        auto _msg = ring_rand(field, {N * n});
        NdArrayView<scalar_t> msg(_msg);
        pforeach(0, msg.numel(), [&](int64_t i) { msg[i] &= mask; });

        std::vector<uint8_t> choices(n);
        std::generate_n(choices.begin(), n, [&]() -> uint8_t {
          return static_cast<uint8_t>(uniform(rdv) % N);
        });

        std::vector<scalar_t> selected(n);

        utils::simulate(kWorldSize,
                        [&](std::shared_ptr<yacl::link::Context> ctx) {
                          auto conn = std::make_shared<Communicator>(ctx);
                          int rank = ctx->Rank();
                          YaclFerretOt ferret(conn, rank == 0, use_ss);
                          size_t sent = ctx->GetStats()->sent_bytes;
                          if (rank == 0) {
                            ferret.SendCMCC(makeConstSpan(msg), N, bw);
                            ferret.Flush();
                          } else {
                            ferret.RecvCMCC(absl::MakeSpan(choices), N,
                                            absl::MakeSpan(selected), bw);
                          }
                          sent = ctx->GetStats()->sent_bytes - sent;
                          SPDLOG_INFO("rank: {}, N: {}, bw: {}, send bytes:{}",
                                      rank, N, bw, sent);
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

class FerretCOTLessLevelTest
    : public testing::TestWithParam<std::tuple<FieldType, bool, int>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, FerretCOTLessLevelTest,
    testing::Combine(testing::Values(FieldType::FM8, FieldType::FM16,
                                     FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(true, false), testing::Values(0, 8, 23)),
    [](const testing::TestParamInfo<FerretCOTLessLevelTest::ParamType> &p) {
      return fmt::format("{}_SS{}_Level{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

// cot, given delta & choice, output (x, x + delta * choice)
// but i-th cot, get \ell-i bits msgs
TEST_P(FerretCOTLessLevelTest, COT_Collapse) {
  size_t kWorldSize = 2;
  int64_t n = 2;
  auto field = std::get<0>(GetParam());
  auto use_ss = std::get<1>(GetParam());
  auto level = std::get<2>(GetParam());

  const auto bw = SizeOf(field) * 8;
  if (level == 0) {
    level = bw / 2;
  }
  if (static_cast<size_t>(level) > bw) {
    return;
  }

  // generate random choices and correlation
  const auto _correlation = ring_rand(field, {static_cast<int64_t>(n * level)});
  // const auto _correlation =
  //     ring_iota(field, {static_cast<int64_t>(n * level)}, 100);
  const auto N = _correlation.numel();

  NdArrayRef oup1 = ring_zeros(field, _correlation.shape());
  NdArrayRef oup2 = ring_zeros(field, _correlation.shape());

  // fixed choices to 1 for debug
  // std::vector<uint8_t> choices(N, 1);

  std::vector<uint8_t> choices(N);
  std::default_random_engine rdv;
  std::uniform_int_distribution<uint64_t> uniform(0, -1);
  std::generate_n(choices.begin(), N, [&]() -> uint8_t {
    return static_cast<uint8_t>(uniform(rdv) & 1);
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;

    auto out1_span = absl::MakeSpan(&oup1.at<u2k>(0), N);
    auto out2_span = absl::MakeSpan(&oup2.at<u2k>(0), N);

    NdArrayView<u2k> correlation(_correlation);

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      int rank = ctx->Rank();

      YaclFerretOt ferret(conn, rank == 0, use_ss);
      if (rank == 0) {
        ferret.SendCAMCC_Collapse(makeConstSpan(correlation), out1_span, bw,
                                  level);
        ferret.Flush();

      } else {
        ferret.RecvCAMCC_Collapse(absl::MakeSpan(choices), out2_span, bw,
                                  level);
      }
    });

    // direct test of COT
    for (int64_t i = 0; i < N; i += n) {
      // Sample-major order
      //      n ||     n     || n         || .... || n
      // k=bw||k=bw - 1||k=bw - 2|| ....
      const auto cur_bw = bw - (i / n);
      const auto mask = makeMask<ring2k_t>(cur_bw);

      for (int64_t j = 0; j < n; ++j) {
        ring2k_t c = (-out1_span[i + j] + out2_span[i + j]) & mask;
        ring2k_t e = (choices[i + j] ? correlation[i + j] : 0) & mask;
        // SPDLOG_INFO("i: {}  mask c: {}", i + j, c);
        // SPDLOG_INFO("i: {} ,e: {}, mask e: {}", i + j,
        //             (choices[i + j] ? correlation[i + j] : 0), e);
        EXPECT_EQ(c, e);
      }
    }

    // first do transpose, then check
    std::vector<u2k> transposed1(N);
    std::vector<u2k> transposed2(N);
    sse_transpose(out1_span.data(), transposed1.data(), level, n);
    sse_transpose(out2_span.data(), transposed2.data(), level, n);

    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < level; ++j) {
        const auto idx = i * level + j;
        const auto transposed_idx = j * n + i;
        const auto cur_bw = bw - j;
        const auto mask = makeMask<ring2k_t>(cur_bw);

        auto c = (-transposed1[idx] + transposed2[idx]) & mask;
        auto e =
            (choices[transposed_idx] ? correlation[transposed_idx] : 0) & mask;

        // SPDLOG_INFO("mask c: {}", c);
        // SPDLOG_INFO("mask e: {}", e);
        EXPECT_EQ(c, e);
      }
    }
  });
}

}  // namespace spu::mpc::cheetah::test
