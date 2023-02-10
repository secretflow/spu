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

#include "libspu/mpc/spdz2k/beaver/beaver_tfp.h"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "yacl/link/link.h"

#include "libspu/core/type_util.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdz2k {

class BeaverTest
    : public ::testing::TestWithParam<
          std::tuple<std::function<std::unique_ptr<BeaverTfpUnsafe>(
                         const std::shared_ptr<yacl::link::Context>& lctx)>,
                     size_t, FieldType, long>> {
 public:
  using Pair = typename BeaverTfpUnsafe::Pair;
  using PairPair = typename BeaverTfpUnsafe::Pair_Pair;
  using TriplePair = typename BeaverTfpUnsafe::Triple_Pair;
};

INSTANTIATE_TEST_SUITE_P(
    BeaverTfpUnsafeTest, BeaverTest,
    testing::Combine(
        testing::Values([](const std::shared_ptr<yacl::link::Context>& lctx) {
          return std::make_unique<BeaverTfpUnsafe>(lctx);
        }),
        testing::Values(4, 3, 2),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(0)),  // max beaver diff,
    [](const testing::TestParamInfo<BeaverTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });

// TODO(@zanxiaopeng.zxp): Add UT for mac and AuthCoinTossing api.
TEST_P(BeaverTest, Mul_large) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t kNumel = 10000;

  std::vector<TriplePair> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx);
                    triples[lctx->Rank()] = beaver->AuthMul(kField, kNumel);
                  });

  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  auto sum_c = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = std::get<0>(triples[r]);
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _a = ArrayView<ring2k_t>(sum_a);
    auto _b = ArrayView<ring2k_t>(sum_b);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < sum_a.numel(); idx++) {
      auto t = _a[idx] * _b[idx];
      auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Mul) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t kNumel = 7;

  std::vector<TriplePair> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx);
                    triples[lctx->Rank()] = beaver->AuthMul(kField, kNumel);
                  });

  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  auto sum_c = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = std::get<0>(triples[r]);
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _a = ArrayView<ring2k_t>(sum_a);
    auto _b = ArrayView<ring2k_t>(sum_b);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < sum_a.numel(); idx++) {
      auto t = _a[idx] * _b[idx];
      auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Dot) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  // M > N
  const size_t M = 17;
  const size_t N = 8;
  const size_t K = 1024;

  std::vector<TriplePair> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx);
                    triples[lctx->Rank()] = beaver->AuthDot(kField, M, N, K);
                  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, M * K);
  auto sum_b = ring_zeros(kField, K * N);
  auto sum_c = ring_zeros(kField, M * N);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = std::get<0>(triples[r]);
    EXPECT_EQ(a.numel(), M * K);
    EXPECT_EQ(b.numel(), K * N);
    EXPECT_EQ(c.numel(), M * N);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  auto res = ring_mmul(sum_a, sum_b, M, N, K);
  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _r = ArrayView<ring2k_t>(res);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < _r.numel(); idx++) {
      auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Dot_large) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  // M < N
  const size_t M = 11;
  const size_t N = 20;
  const size_t K = 1023;

  std::vector<TriplePair> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx);
                    triples[lctx->Rank()] = beaver->AuthDot(kField, M, N, K);
                  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, M * K);
  auto sum_b = ring_zeros(kField, K * N);
  auto sum_c = ring_zeros(kField, M * N);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = std::get<0>(triples[r]);
    EXPECT_EQ(a.numel(), M * K);
    EXPECT_EQ(b.numel(), K * N);
    EXPECT_EQ(c.numel(), M * N);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  auto res = ring_mmul(sum_a, sum_b, M, N, K);
  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _r = ArrayView<ring2k_t>(res);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < _r.numel(); idx++) {
      auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Trunc) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;
  const size_t kBits = 5;

  std::vector<PairPair> pairs;
  pairs.resize(kWorldSize);

  utils::simulate(
      kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        auto beaver = factory(lctx);
        pairs[lctx->Rank()] = beaver->AuthTrunc(kField, kNumel, kBits);
      });

  EXPECT_EQ(pairs.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b] = std::get<0>(pairs[r]);
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
  }
  EXPECT_EQ(ring_arshift(sum_a, kBits), sum_b) << sum_a << sum_b;
}

}  // namespace spu::mpc::spdz2k
